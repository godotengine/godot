//===- llvm/Support/Windows/MSFileSystem.cpp - DXCompiler Impl --*- C++ -*-===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// MSFileSystem.inc.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file implements the Windows specific implementation of the Path API. //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/ADT/STLExtras.h"
#include "WindowsSupport.h"
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include "llvm/Support/WindowsError.h"
#include "llvm/Support/MSFileSystem.h"

// MinGW doesn't define this.
#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

using namespace llvm;

using std::error_code;
using std::system_category;

using llvm::sys::windows::UTF8ToUTF16;
using llvm::sys::windows::UTF16ToUTF8;
using llvm::sys::path::widenPath;

namespace llvm {
namespace sys  {

namespace windows {
  error_code UTF8ToUTF16(llvm::StringRef utf8,
    llvm::SmallVectorImpl<wchar_t> &utf16);
  error_code UTF16ToUTF8(const wchar_t *utf16, size_t utf16_len,
    llvm::SmallVectorImpl<char> &utf8);
} // llvm::sys::windows

namespace fs {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Per-thread MSFileSystem support.

namespace {

template <typename _T>
class ThreadLocalStorage {
  DWORD m_Tls;
  DWORD m_dwError;
public:
  ThreadLocalStorage() : m_Tls(TLS_OUT_OF_INDEXES), m_dwError(ERROR_NOT_READY) {}
  DWORD Setup() {
    if (m_Tls == TLS_OUT_OF_INDEXES) {
      m_Tls = TlsAlloc();
      m_dwError = (m_Tls == TLS_OUT_OF_INDEXES) ? ::GetLastError() : 0;
    }
    return m_dwError;
  }
  void Cleanup() {
    if (m_Tls != TLS_OUT_OF_INDEXES)
      TlsFree(m_Tls);
    m_Tls = TLS_OUT_OF_INDEXES;
    m_dwError = ERROR_NOT_READY;
  }
  ~ThreadLocalStorage() { Cleanup(); }
  _T GetValue() const {
    if (m_Tls != TLS_OUT_OF_INDEXES)
      return (_T)TlsGetValue(m_Tls);
    else
      return nullptr;
  }
  bool SetValue(_T value) {
    if (m_Tls != TLS_OUT_OF_INDEXES) {
      return TlsSetValue(m_Tls, (void*)value);
    } else {
      ::SetLastError(m_dwError);
      return false;
    }
  }
  // Retrieve error code if TlsAlloc() failed
  DWORD GetError() const {
    return m_dwError;
  }
  operator bool() const { return m_Tls != TLS_OUT_OF_INDEXES; }
};

static ThreadLocalStorage<MSFileSystemRef> g_PerThreadSystem;

}

error_code GetFileSystemTlsStatus() throw() {
  DWORD dwError = g_PerThreadSystem.GetError();
  if (dwError)
    return error_code(dwError, system_category());
  else
    return error_code();
}

error_code SetupPerThreadFileSystem() throw() {
  assert(!g_PerThreadSystem && g_PerThreadSystem.GetError() == ERROR_NOT_READY &&
          "otherwise, PerThreadSystem already set up.");
  if (g_PerThreadSystem.Setup())
    return GetFileSystemTlsStatus();
  return error_code();
}
void CleanupPerThreadFileSystem() throw() {
  g_PerThreadSystem.Cleanup();
}

MSFileSystemRef GetCurrentThreadFileSystem() throw() {
#ifdef MS_IMPLICIT_DISK_FILESYSTEM
  if (!g_PerThreadSystem)
    getImplicitFilesystem();
#endif

  assert(g_PerThreadSystem && "otherwise, TLS not initialized");
  return g_PerThreadSystem.GetValue();
}

error_code SetCurrentThreadFileSystem(MSFileSystemRef value) throw()
{
  assert(g_PerThreadSystem && "otherwise, TLS not initialized");
  // For now, disallow reentrancy in APIs (i.e., replace the current instance with another one).
  if (value != nullptr)
  {
    MSFileSystemRef current = GetCurrentThreadFileSystem();
    if (current != nullptr)
    {
      return error_code(ERROR_POSSIBLE_DEADLOCK, system_category());
    }
  }

  if (!g_PerThreadSystem.SetValue(value))
  {
    return mapWindowsError(::GetLastError());
  }

  return error_code();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Support for CRT-like file stream functions.

int msf_read(int fd, void* buffer, unsigned int count) throw()
{
  MSFileSystemRef fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->Read(fd, buffer, count);
}

int msf_write(int fd, const void* buffer, unsigned int count) throw()
{
  MSFileSystemRef fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->Write(fd, buffer, count);
}

int msf_close(int fd) throw()
{
  MSFileSystemRef fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->close(fd);
}

long msf_lseek(int fd, long offset, int origin)
{
  MSFileSystemRef fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->lseek(fd, offset, origin);
}

int msf_setmode(int fd, int mode) throw()
{
  MSFileSystemRef fsr = GetCurrentThreadFileSystem();
  if (fsr == nullptr) {
    errno = EBADF;
    return -1;
  }
  return fsr->setmode(fd, mode);
}

} } }

///////////////////////////////////////////////////////////////////////////////////////////////////
// MSFileSystem-based support for Path APIs.

typedef llvm::sys::fs::MSFileSystemRef MSFileSystemRef;

static
error_code GetCurrentThreadFileSystemOrError(_Outptr_ MSFileSystemRef* pResult) throw()
{
  *pResult = ::llvm::sys::fs::GetCurrentThreadFileSystem();

  // It is an error to have an I/O API invoked without having installed support 
  // for it. We handle it gracefully in case there is problem while shutting 
  // down, but this is a bug that should be fixed.
  assert(*pResult);

  if (*pResult == nullptr) {
    return mapWindowsError(ERROR_NOT_READY);
  }

  return error_code();
}

static bool is_separator(const wchar_t value) {
  switch (value) {
  case L'\\':
  case L'/':
    return true;
  default:
    return false;
  }
}

// TODO: consider erasing this
namespace {
  error_code TempDir(_In_ MSFileSystemRef fsr, SmallVectorImpl<wchar_t> &result) {
  retry_temp_dir:
    DWORD len = fsr->GetTempPathW(result.capacity(), result.begin());

    if (len == 0)
      return mapWindowsError(::GetLastError());

    if (len > result.capacity()) {
      result.reserve(len);
      goto retry_temp_dir;
    }

    result.set_size(len);
    return error_code();
  }
}

namespace llvm {
namespace sys  {

namespace path {
  // Convert a UTF-8 path to UTF-16.  Also, if the absolute equivalent of the
  // path is longer than CreateDirectory can tolerate, make it absolute and
  // prefixed by '\\?\'.
  std::error_code widenPath(const Twine &Path8,
    SmallVectorImpl<wchar_t> &Path16) {
    const size_t MaxDirLen = MAX_PATH - 12; // Must leave room for 8.3 filename.

    // Several operations would convert Path8 to SmallString; more efficient to
    // do it once up front.
    SmallString<128> Path8Str;
    Path8.toVector(Path8Str);

    // If we made this path absolute, how much longer would it get?
    size_t CurPathLen;
    if (llvm::sys::path::is_absolute(Twine(Path8Str)))
      CurPathLen = 0; // No contribution from current_path needed.
    else {
      CurPathLen = ::GetCurrentDirectoryW(0, NULL);
      if (CurPathLen == 0)
        return mapWindowsError(::GetLastError());
    }

    // Would the absolute path be longer than our limit?
    if ((Path8Str.size() + CurPathLen) >= MaxDirLen &&
      !Path8Str.startswith("\\\\?\\")) {
      SmallString<2 * MAX_PATH> FullPath("\\\\?\\");
      if (CurPathLen) {
        SmallString<80> CurPath;
        if (std::error_code EC = llvm::sys::fs::current_path(CurPath))
          return EC;
        FullPath.append(CurPath);
      }
      // Traverse the requested path, canonicalizing . and .. as we go (because
      // the \\?\ prefix is documented to treat them as real components).
      // The iterators don't report separators and append() always attaches
      // preferred_separator so we don't need to call native() on the result.
      for (llvm::sys::path::const_iterator I = llvm::sys::path::begin(Path8Str),
        E = llvm::sys::path::end(Path8Str);
        I != E; ++I) {
        if (I->size() == 1 && *I == ".")
          continue;
        if (I->size() == 2 && *I == "..")
          llvm::sys::path::remove_filename(FullPath);
        else
          llvm::sys::path::append(FullPath, *I);
      }
      return UTF8ToUTF16(FullPath, Path16);
    }

    // Just use the caller's original path.
    return UTF8ToUTF16(Path8Str, Path16);
  }

bool home_directory(SmallVectorImpl<char> &result) {
  assert("HLSL Unimplemented!");
  return false;
}

} // end namespace path

namespace fs {

std::string getMainExecutable(const char *argv0, void *MainExecAddr) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return "";

  SmallVector<wchar_t, MAX_PATH> PathName;
  DWORD Size = fsr->GetMainModuleFileNameW(PathName.data(), PathName.capacity());

  // A zero return value indicates a failure other than insufficient space.
  if (Size == 0)
    return "";

  // Insufficient space is determined by a return value equal to the size of
  // the buffer passed in.
  if (Size == PathName.capacity())
    return "";

  // On success, GetModuleFileNameW returns the number of characters written to
  // the buffer not including the NULL terminator.
  PathName.set_size(Size);

  // Convert the result from UTF-16 to UTF-8.
  SmallVector<char, MAX_PATH> PathNameUTF8;
  if (UTF16ToUTF8(PathName.data(), PathName.size(), PathNameUTF8))
    return "";

  return std::string(PathNameUTF8.data());
}

UniqueID file_status::getUniqueID() const {
  // The file is uniquely identified by the volume serial number along
  // with the 64-bit file identifier.
  uint64_t FileID = (static_cast<uint64_t>(FileIndexHigh) << 32ULL) |
                    static_cast<uint64_t>(FileIndexLow);

  return UniqueID(VolumeSerialNumber, FileID);
}

TimeValue file_status::getLastModificationTime() const {
  ULARGE_INTEGER UI;
  UI.LowPart = LastWriteTimeLow;
  UI.HighPart = LastWriteTimeHigh;

  TimeValue Ret;
  Ret.fromWin32Time(UI.QuadPart);
  return Ret;
}

error_code current_path(SmallVectorImpl<char> &result) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallVector<wchar_t, MAX_PATH> cur_path;
  DWORD len = MAX_PATH;

  do {
    cur_path.reserve(len);
    len = fsr->GetCurrentDirectoryW(cur_path.capacity(), cur_path.data());

    // A zero return value indicates a failure other than insufficient space.
    if (len == 0)
      return mapWindowsError(::GetLastError());

    // If there's insufficient space, the len returned is larger than the len
    // given.
  } while (len > cur_path.capacity());

  // On success, GetCurrentDirectoryW returns the number of characters not
  // including the null-terminator.
  cur_path.set_size(len);
  return UTF16ToUTF8(cur_path.begin(), cur_path.size(), result);
}

std::error_code create_directory(const Twine &path, bool IgnoreExisting) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage), path_utf16))
    return ec;

  if (!fsr->CreateDirectoryW(path_utf16.begin())) {
    DWORD LastError = ::GetLastError();
    if (LastError != ERROR_ALREADY_EXISTS || !IgnoreExisting)
      return mapWindowsError(LastError);
  }

  return std::error_code();
}

// We can't use symbolic links for windows.
std::error_code create_link(const Twine &to, const Twine &from) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;
  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (std::error_code ec = widenPath(from, wide_from))
    return ec;
  if (std::error_code ec = widenPath(to, wide_to))
    return ec;

  if (!fsr->CreateHardLinkW(wide_from.begin(), wide_to.begin()))
    return mapWindowsError(::GetLastError());

  return std::error_code();
}

std::error_code remove(const Twine &path, bool IgnoreNonExisting) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallVector<wchar_t, 128> path_utf16;

  file_status ST;
  if (std::error_code EC = status(path, ST)) {
    if (EC != errc::no_such_file_or_directory || !IgnoreNonExisting)
      return EC;
    return std::error_code();
  }

  if (std::error_code ec = widenPath(path, path_utf16))
    return ec;

  if (ST.type() == file_type::directory_file) {
    if (!fsr->RemoveDirectoryW(c_str(path_utf16))) {
      std::error_code EC = mapWindowsError(::GetLastError());
      if (EC != errc::no_such_file_or_directory || !IgnoreNonExisting)
        return EC;
    }
    return std::error_code();
  }
  if (!fsr->DeleteFileW(c_str(path_utf16))) {
    std::error_code EC = mapWindowsError(::GetLastError());
    if (EC != errc::no_such_file_or_directory || !IgnoreNonExisting)
      return EC;
  }
  return std::error_code();
}

error_code rename(const Twine &from, const Twine &to) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (error_code ec = UTF8ToUTF16(f, wide_from)) return ec;
  if (error_code ec = UTF8ToUTF16(t, wide_to)) return ec;

  error_code ec = error_code();
  for (int i = 0; i < 2000; i++) {
    if (fsr->MoveFileExW(wide_from.begin(), wide_to.begin(),
                      MOVEFILE_COPY_ALLOWED | MOVEFILE_REPLACE_EXISTING))
      return error_code();
    ec = mapWindowsError(::GetLastError());
    if (ec != std::errc::permission_denied)
      break;
    // Retry MoveFile() at ACCESS_DENIED.
    // System scanners (eg. indexer) might open the source file when
    // It is written and closed.
    ::Sleep(1);
  }

  return ec;
}

error_code resize_file(const Twine &path, uint64_t size) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  return error_code(fsr->resize_file(path_utf16.begin(), size), std::generic_category());
}

error_code exists(const Twine &path, bool &result) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  DWORD attributes = fsr->GetFileAttributesW(path_utf16.begin());

  if (attributes == INVALID_FILE_ATTRIBUTES) {
    // See if the file didn't actually exist.
    error_code ec = mapWindowsError(::GetLastError());
    if (ec != std::errc::no_such_file_or_directory)
      return ec;
    result = false;
  } else
    result = true;
  return error_code();
}

std::error_code access(const Twine &Path, AccessMode Mode) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> PathStorage;
  SmallVector<wchar_t, 128> PathUtf16;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);
  if (error_code ec = UTF8ToUTF16(P, PathUtf16))
    return ec;

  DWORD Attr = fsr->GetFileAttributesW(PathUtf16.begin());
  // TODO: look at GetLastError as well.
  if (Attr == INVALID_FILE_ATTRIBUTES) {
    return make_error_code(std::errc::no_such_file_or_directory);
  }

  switch (Mode) {
  case AccessMode::Exist:
  case AccessMode::Execute:
    // Consider: directories should not be executable.
    return std::error_code();
  default:
    assert(Mode == AccessMode::Write && "no other enum value allowed");
  case AccessMode::Write:
    return !(Attr & FILE_ATTRIBUTE_READONLY) ?
      std::error_code() : make_error_code(std::errc::permission_denied);
  }
}

bool equivalent(file_status A, file_status B) {
  assert(status_known(A) && status_known(B));
  return A.FileIndexHigh      == B.FileIndexHigh &&
         A.FileIndexLow       == B.FileIndexLow &&
         A.FileSizeHigh       == B.FileSizeHigh &&
         A.FileSizeLow        == B.FileSizeLow &&
         A.LastWriteTimeHigh  == B.LastWriteTimeHigh &&
         A.LastWriteTimeLow   == B.LastWriteTimeLow &&
         A.VolumeSerialNumber == B.VolumeSerialNumber;
}

error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (error_code ec = status(A, fsA)) return ec;
  if (error_code ec = status(B, fsB)) return ec;
  result = equivalent(fsA, fsB);
  return error_code();
}

static bool isReservedName(StringRef path) {
  // This list of reserved names comes from MSDN, at:
  // http://msdn.microsoft.com/en-us/library/aa365247%28v=vs.85%29.aspx
  static const char *sReservedNames[] = { "nul", "con", "prn", "aux",
                              "com1", "com2", "com3", "com4", "com5", "com6",
                              "com7", "com8", "com9", "lpt1", "lpt2", "lpt3",
                              "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9" };

  // First, check to see if this is a device namespace, which always
  // starts with \\.\, since device namespaces are not legal file paths.
  if (path.startswith("\\\\.\\"))
    return true;

  // Then compare against the list of ancient reserved names
  for (size_t i = 0; i < array_lengthof(sReservedNames); ++i) {
    if (path.equals_lower(sReservedNames[i]))
      return true;
  }

  // The path isn't what we consider reserved.
  return false;
}

static error_code getStatus(HANDLE FileHandle, file_status &Result) {
  if (FileHandle == INVALID_HANDLE_VALUE)
    goto handle_status_error;

  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  switch (fsr->GetFileType(FileHandle)) {
  default:
    llvm_unreachable("Don't know anything about this file type");
  case FILE_TYPE_UNKNOWN: {
    DWORD Err = ::GetLastError();
    if (Err != NO_ERROR)
      return mapWindowsError(Err);
    Result = file_status(file_type::type_unknown);
    return error_code();
  }
  case FILE_TYPE_DISK:
    break;
  case FILE_TYPE_CHAR:
    Result = file_status(file_type::character_file);
    return error_code();
  case FILE_TYPE_PIPE:
    Result = file_status(file_type::fifo_file);
    return error_code();
  }

  BY_HANDLE_FILE_INFORMATION Info;
  if (!fsr->GetFileInformationByHandle(FileHandle, &Info))
    goto handle_status_error;

  {
    file_type Type = (Info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                         ? file_type::directory_file
                         : file_type::regular_file;
    Result =
        file_status(Type, Info.ftLastWriteTime.dwHighDateTime,
                    Info.ftLastWriteTime.dwLowDateTime,
                    Info.dwVolumeSerialNumber, Info.nFileSizeHigh,
                    Info.nFileSizeLow, Info.nFileIndexHigh, Info.nFileIndexLow);
    return error_code();
  }

handle_status_error:
  error_code EC = mapWindowsError(::GetLastError());
  if (EC == std::errc::no_such_file_or_directory)
    Result = file_status(file_type::file_not_found);
  else
    Result = file_status(file_type::status_error);
  return EC;
}

error_code status(const Twine &path, file_status &result) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  StringRef path8 = path.toStringRef(path_storage);
  if (isReservedName(path8)) {
    result = file_status(file_type::character_file);
    return error_code();
  }

  if (error_code ec = UTF8ToUTF16(path8, path_utf16))
    return ec;

  DWORD attr = fsr->GetFileAttributesW(path_utf16.begin());
  if (attr == INVALID_FILE_ATTRIBUTES)
    return getStatus(INVALID_HANDLE_VALUE, result);

  // Handle reparse points.
  if (attr & FILE_ATTRIBUTE_REPARSE_POINT) {
    ScopedFileHandle h(
      fsr->CreateFileW(path_utf16.begin(),
                    0, // Attributes only.
                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS));
    if (!h)
      return getStatus(INVALID_HANDLE_VALUE, result);
  }

  ScopedFileHandle h(
      fsr->CreateFileW(path_utf16.begin(), 0, // Attributes only.
                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                    OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS));
    if (!h)
      return getStatus(INVALID_HANDLE_VALUE, result);

    return getStatus(h, result);
}

error_code status(int FD, file_status &Result) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  HANDLE FileHandle = reinterpret_cast<HANDLE>(fsr->get_osfhandle(FD));
  return getStatus(FileHandle, Result);
}

error_code setLastModificationAndAccessTime(int FD, TimeValue Time) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  ULARGE_INTEGER UI;
  UI.QuadPart = Time.toWin32Time();
  FILETIME FT;
  FT.dwLowDateTime = UI.LowPart;
  FT.dwHighDateTime = UI.HighPart;
  HANDLE FileHandle = reinterpret_cast<HANDLE>(fsr->get_osfhandle(FD));
  if (!fsr->SetFileTime(FileHandle, NULL, &FT, &FT))
    return mapWindowsError(::GetLastError());
  return error_code();
}

error_code get_magic(const Twine &path, uint32_t len,
                     SmallVectorImpl<char> &result) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;
  result.set_size(0);

  // Convert path to UTF-16.
  if (error_code ec = UTF8ToUTF16(path.toStringRef(path_storage),
                                  path_utf16))
    return ec;

  // Open file.
  HANDLE file = fsr->CreateFileW(c_str(path_utf16),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_READONLY);
  if (file == INVALID_HANDLE_VALUE)
    return mapWindowsError(::GetLastError());

  // Allocate buffer.
  result.reserve(len);

  // Get magic!
  DWORD bytes_read = 0;
  BOOL read_success = fsr->ReadFile(file, result.data(), len, &bytes_read);
  error_code ec = mapWindowsError(::GetLastError());
  fsr->CloseHandle(file);
  if (!read_success || (bytes_read != len)) {
    // Set result size to the number of bytes read if it's valid.
    if (bytes_read <= len)
      result.set_size(bytes_read);
    // ERROR_HANDLE_EOF is mapped to errc::value_too_large.
    return ec;
  }

  result.set_size(len);
  return error_code();
}

error_code mapped_file_region::init(int FD, uint64_t Offset, mapmode Mode) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  // Make sure that the requested size fits within SIZE_T.
  if (Size > std::numeric_limits<SIZE_T>::max())
    return make_error_code(errc::invalid_argument);

  HANDLE FileHandle = reinterpret_cast<HANDLE>(fsr->get_osfhandle(FD));
  if (FileHandle == INVALID_HANDLE_VALUE)
    return make_error_code(errc::bad_file_descriptor);

  DWORD flprotect;
  switch (Mode) {
  case readonly:  flprotect = PAGE_READONLY; break;
  case readwrite: flprotect = PAGE_READWRITE; break;
  case priv:      flprotect = PAGE_WRITECOPY; break;
  }

  HANDLE FileMappingHandle =
    fsr->CreateFileMappingW(FileHandle, flprotect,
    (Offset + Size) >> 32,
    (Offset + Size) & 0xffffffff);
  if (FileMappingHandle == NULL) {
    std::error_code ec = mapWindowsError(GetLastError());
    return ec;
  }

  DWORD dwDesiredAccess;
  switch (Mode) {
  case readonly:  dwDesiredAccess = FILE_MAP_READ; break;
  case readwrite: dwDesiredAccess = FILE_MAP_WRITE; break;
  case priv:      dwDesiredAccess = FILE_MAP_COPY; break;
  }
  Mapping = fsr->MapViewOfFile(FileMappingHandle,
    dwDesiredAccess,
    Offset >> 32,
    Offset & 0xffffffff,
    Size);
  if (Mapping == NULL) {
    std::error_code ec = mapWindowsError(GetLastError());
    fsr->CloseHandle(FileMappingHandle);
    return ec;
  }

  if (Size == 0) {
    MEMORY_BASIC_INFORMATION mbi;
    SIZE_T Result = VirtualQuery(Mapping, &mbi, sizeof(mbi)); // TODO: do we need to plumb through fsr?
    if (Result == 0) {
      std::error_code ec = mapWindowsError(GetLastError());
      fsr->UnmapViewOfFile(Mapping);
      fsr->CloseHandle(FileMappingHandle);
      return ec;
    }
    Size = mbi.RegionSize;
  }

  // Close all the handles except for the view. It will keep the other handles
  // alive.
  fsr->CloseHandle(FileMappingHandle);
  return std::error_code();
}


mapped_file_region::mapped_file_region(int fd, mapmode mode, uint64_t length,
  uint64_t offset, std::error_code &ec)
  : Size(length), Mapping() {
  ec = init(fd, offset, mode);
  if (ec)
    Mapping = 0;
}

mapped_file_region::~mapped_file_region() {
  if (Mapping)
    GetCurrentThreadFileSystem()->UnmapViewOfFile(Mapping);
}

uint64_t mapped_file_region::size() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return Size;
}

char *mapped_file_region::data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<char*>(Mapping);
}

const char *mapped_file_region::const_data() const {
  assert(Mapping && "Mapping failed but used anyway!");
  return reinterpret_cast<const char*>(Mapping);
}

int mapped_file_region::alignment() {
  SYSTEM_INFO SysInfo;
  ::GetSystemInfo(&SysInfo);
  return SysInfo.dwAllocationGranularity;
}

error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallVector<wchar_t, 128> path_utf16;

  if (error_code ec = UTF8ToUTF16(path,
                                  path_utf16))
    return ec;

  // Convert path to the format that Windows is happy with.
  if (path_utf16.size() > 0 &&
      !is_separator(path_utf16[path.size() - 1]) &&
      path_utf16[path.size() - 1] != L':') {
    path_utf16.push_back(L'\\');
    path_utf16.push_back(L'*');
  } else {
    path_utf16.push_back(L'*');
  }

  //  Get the first directory entry.
  WIN32_FIND_DATAW FirstFind;
  ScopedFindHandle FindHandle(fsr->FindFirstFileW(c_str(path_utf16), &FirstFind));
  if (!FindHandle)
    return mapWindowsError(::GetLastError());

  size_t FilenameLen = ::wcslen(FirstFind.cFileName);
  while ((FilenameLen == 1 && FirstFind.cFileName[0] == L'.') ||
         (FilenameLen == 2 && FirstFind.cFileName[0] == L'.' &&
                              FirstFind.cFileName[1] == L'.'))
    if (!fsr->FindNextFileW(FindHandle, &FirstFind)) {
      DWORD lastError = ::GetLastError();
      // Check for end.
      if (lastError == ERROR_NO_MORE_FILES) // no more files
        return detail::directory_iterator_destruct(it);
      return mapWindowsError(lastError);
    } else
      FilenameLen = ::wcslen(FirstFind.cFileName);

  // Construct the current directory entry.
  SmallString<128> directory_entry_name_utf8;
  if (error_code ec = UTF16ToUTF8(FirstFind.cFileName,
                                  ::wcslen(FirstFind.cFileName),
                                  directory_entry_name_utf8))
    return ec;

  it.IterationHandle = intptr_t(FindHandle.take());
  SmallString<128> directory_entry_path(path);
  path::append(directory_entry_path, directory_entry_name_utf8.str());
  it.CurrentEntry = directory_entry(directory_entry_path.str());

  return error_code();
}

error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle != 0)
    // Closes the handle if it's valid.
    ScopedFindHandle close(HANDLE(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return error_code();
}

error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  WIN32_FIND_DATAW FindData;
  if (!fsr->FindNextFileW(HANDLE(it.IterationHandle), &FindData)) {
    DWORD lastError = ::GetLastError();
    // Check for end.
    if (lastError == ERROR_NO_MORE_FILES) // no more files
      return detail::directory_iterator_destruct(it);
    return mapWindowsError(lastError);
  }

  size_t FilenameLen = ::wcslen(FindData.cFileName);
  if ((FilenameLen == 1 && FindData.cFileName[0] == L'.') ||
      (FilenameLen == 2 && FindData.cFileName[0] == L'.' &&
                           FindData.cFileName[1] == L'.'))
    return directory_iterator_increment(it);

  SmallString<128> directory_entry_path_utf8;
  if (error_code ec = UTF16ToUTF8(FindData.cFileName,
                                  ::wcslen(FindData.cFileName),
                                  directory_entry_path_utf8))
    return ec;

  it.CurrentEntry.replace_filename(Twine(directory_entry_path_utf8));
  return error_code();
}

error_code openFileForRead(const Twine &Name, int &ResultFD) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  SmallString<128> PathStorage;
  SmallVector<wchar_t, 128> PathUTF16;

  if (error_code EC = UTF8ToUTF16(Name.toStringRef(PathStorage),
                                  PathUTF16))
    return EC;

  HANDLE H = fsr->CreateFileW(PathUTF16.begin(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
  if (H == INVALID_HANDLE_VALUE) {
    DWORD LastError = ::GetLastError();
    std::error_code EC = mapWindowsError(LastError);
    // Provide a better error message when trying to open directories.
    // This only runs if we failed to open the file, so there is probably
    // no performances issues.
    if (LastError != ERROR_ACCESS_DENIED)
      return EC;
    if (is_directory(Name))
      return make_error_code(errc::is_a_directory);
    return EC;
  }

  int FD = fsr->open_osfhandle(intptr_t(H), 0);
  if (FD == -1) {
    fsr->CloseHandle(H);
    return mapWindowsError(ERROR_INVALID_HANDLE);
  }

  ResultFD = FD;
  return error_code();
}

error_code openFileForWrite(const Twine &Name, int &ResultFD,
                            sys::fs::OpenFlags Flags, unsigned Mode) {
  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return ec;

  // Verify that we don't have both "append" and "excl".
  assert((!(Flags & sys::fs::F_Excl) || !(Flags & sys::fs::F_Append)) &&
         "Cannot specify both 'excl' and 'append' file creation flags!");

  SmallString<128> PathStorage;
  SmallVector<wchar_t, 128> PathUTF16;

  if (error_code EC = UTF8ToUTF16(Name.toStringRef(PathStorage),
                                  PathUTF16))
    return EC;

  DWORD CreationDisposition;
  if (Flags & F_Excl)
    CreationDisposition = CREATE_NEW;
  else if (Flags & F_Append)
    CreationDisposition = OPEN_ALWAYS;
  else
    CreationDisposition = CREATE_ALWAYS;

  HANDLE H = fsr->CreateFileW(PathUTF16.begin(), GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           CreationDisposition, FILE_ATTRIBUTE_NORMAL);

  if (H == INVALID_HANDLE_VALUE) {
    DWORD LastError = ::GetLastError();
    std::error_code EC = mapWindowsError(LastError);
    // Provide a better error message when trying to open directories.
    // This only runs if we failed to open the file, so there is probably
    // no performances issues.
    if (LastError != ERROR_ACCESS_DENIED)
      return EC;
    if (is_directory(Name))
      return make_error_code(errc::is_a_directory);
    return EC;
  }

  int OpenFlags = 0;
  if (Flags & F_Append)
    OpenFlags |= _O_APPEND;

  if (Flags & F_Text)
    OpenFlags |= _O_TEXT;

  int FD = fsr->open_osfhandle(intptr_t(H), OpenFlags);
  if (FD == -1) {
    fsr->CloseHandle(H);
    return mapWindowsError(ERROR_INVALID_HANDLE);
  }

  ResultFD = FD;
  return error_code();
}

std::error_code resize_file(int FD, uint64_t Size) {
#ifdef HAVE__CHSIZE_S
  errno_t error = ::_chsize_s(FD, Size);
#else
  errno_t error = ::_chsize(FD, Size);
#endif
  return std::error_code(error, std::generic_category());
}

} // end namespace fs

namespace path {

void system_temp_directory(bool ErasedOnReboot, SmallVectorImpl<char> &Result) {
  (void)ErasedOnReboot;
  Result.clear();

  MSFileSystemRef fsr;
  if (error_code ec = GetCurrentThreadFileSystemOrError(&fsr)) return;

  SmallVector<wchar_t, 128> result;
  DWORD len;

retry_temp_dir:
  len = fsr->GetTempPathW(result.capacity(), result.begin());

  if (len == 0)
    return; // mapWindowsError(::GetLastError());

  if (len > result.capacity()) {
    result.reserve(len);
    goto retry_temp_dir;
  }

  result.set_size(len);
  
  UTF16ToUTF8(result.begin(), result.size(), Result);
}

} // end namespace path

namespace windows {
std::error_code ACPToUTF16(llvm::StringRef acp,
                            llvm::SmallVectorImpl<wchar_t> &utf16) {
  int len = ::MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS,
                                  acp.begin(), acp.size(),
                                  utf16.begin(), 0);

  if (len == 0)
    return mapWindowsError(::GetLastError());

  utf16.reserve(len + 1);
  utf16.set_size(len);

  len = ::MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS,
                              acp.begin(), acp.size(),
                              utf16.begin(), utf16.size());

  if (len == 0)
    return mapWindowsError(::GetLastError());

  // Make utf16 null terminated.
  utf16.push_back(0);
  utf16.pop_back();

  return error_code();
}

std::error_code ACPToUTF8(const char *acp, size_t acp_len,
                           llvm::SmallVectorImpl<char> &utf8) {
  llvm::SmallVector<wchar_t, 128> utf16;
  std::error_code ec = ACPToUTF16(StringRef(acp, acp_len), utf16);
  if (ec) return ec;
  return UTF16ToUTF8(utf16.begin(), utf16.size(), utf8);
}

std::error_code UTF8ToUTF16(llvm::StringRef utf8,
                             llvm::SmallVectorImpl<wchar_t> &utf16) {
  int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                  utf8.begin(), utf8.size(),
                                  utf16.begin(), 0);

  if (len == 0)
    return mapWindowsError(::GetLastError());

  utf16.reserve(len + 1);
  utf16.set_size(len);

  len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                              utf8.begin(), utf8.size(),
                              utf16.begin(), utf16.size());

  if (len == 0)
    return mapWindowsError(::GetLastError());

  // Make utf16 null terminated.
  utf16.push_back(0);
  utf16.pop_back();

  return error_code();
}

static
std::error_code UTF16ToCodePage(unsigned codepage, const wchar_t *utf16,
                                size_t utf16_len,
                                llvm::SmallVectorImpl<char> &utf8) {
  if (utf16_len) {
    // Get length.
    int len = ::WideCharToMultiByte(codepage, 0, utf16, utf16_len, utf8.begin(),
                                    0, NULL, NULL);

    if (len == 0)
      return mapWindowsError(::GetLastError());

    utf8.reserve(len);
    utf8.set_size(len);

    // Now do the actual conversion.
    len = ::WideCharToMultiByte(codepage, 0, utf16, utf16_len, utf8.data(),
                                utf8.size(), NULL, NULL);

    if (len == 0)
      return mapWindowsError(::GetLastError());
  }

  // Make utf8 null terminated.
  utf8.push_back(0);
  utf8.pop_back();

  return std::error_code();
}

std::error_code UTF16ToUTF8(const wchar_t *utf16, size_t utf16_len,
  llvm::SmallVectorImpl<char> &utf8) {
  return UTF16ToCodePage(CP_UTF8, utf16, utf16_len, utf8);
}

std::error_code UTF16ToCurCP(const wchar_t *utf16, size_t utf16_len,
  llvm::SmallVectorImpl<char> &utf8) {
  return UTF16ToCodePage(CP_ACP, utf16, utf16_len, utf8);
}

} // end namespace windows
} // end namespace sys
} // end namespace llvm
