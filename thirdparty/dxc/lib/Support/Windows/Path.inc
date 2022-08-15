//===- llvm/Support/Windows/Path.inc - Windows Path Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Windows specific implementation of the Path API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Windows code that
//===          is guaranteed to work on *all* Windows variants.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/WindowsError.h"
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>

// These two headers must be included last, and make sure shlobj is required
// after Windows.h to make sure it picks up our definition of _WIN32_WINNT
#include "WindowsSupport.h"
// #include <shlobj.h> // HLSL Change - no support for writing to home directory

#undef max

// MinGW doesn't define this.
#ifndef _ERRNO_T_DEFINED
#define _ERRNO_T_DEFINED
typedef int errno_t;
#endif

#if 0 // HLSL Change Starts
#ifdef _MSC_VER
# pragma comment(lib, "advapi32.lib")  // This provides CryptAcquireContextW.
#endif
#endif // HLSL Change Ends

using namespace llvm;

using llvm::sys::windows::UTF8ToUTF16;
using llvm::sys::windows::UTF16ToUTF8;
using llvm::sys::path::widenPath;

static bool is_separator(const wchar_t value) {
  switch (value) {
  case L'\\':
  case L'/':
    return true;
  default:
    return false;
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
    SmallString<2*MAX_PATH> FullPath("\\\\?\\");
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
} // end namespace path

namespace fs {

std::string getMainExecutable(const char *argv0, void *MainExecAddr) {
  SmallVector<wchar_t, MAX_PATH> PathName;
  DWORD Size = ::GetModuleFileNameW(NULL, PathName.data(), PathName.capacity());

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

std::error_code current_path(SmallVectorImpl<char> &result) {
  SmallVector<wchar_t, MAX_PATH> cur_path;
  DWORD len = MAX_PATH;

  do {
    cur_path.reserve(len);
    len = ::GetCurrentDirectoryW(cur_path.capacity(), cur_path.data());

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
  SmallVector<wchar_t, 128> path_utf16;

  if (std::error_code ec = widenPath(path, path_utf16))
    return ec;

  if (!::CreateDirectoryW(path_utf16.begin(), NULL)) {
    DWORD LastError = ::GetLastError();
    if (LastError != ERROR_ALREADY_EXISTS || !IgnoreExisting)
      return mapWindowsError(LastError);
  }

  return std::error_code();
}

// We can't use symbolic links for windows.
std::error_code create_link(const Twine &to, const Twine &from) {
  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (std::error_code ec = widenPath(from, wide_from))
    return ec;
  if (std::error_code ec = widenPath(to, wide_to))
    return ec;

  if (!::CreateHardLinkW(wide_from.begin(), wide_to.begin(), NULL))
    return mapWindowsError(::GetLastError());

  return std::error_code();
}

std::error_code remove(const Twine &path, bool IgnoreNonExisting) {
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
    if (!::RemoveDirectoryW(c_str(path_utf16))) {
      std::error_code EC = mapWindowsError(::GetLastError());
      if (EC != errc::no_such_file_or_directory || !IgnoreNonExisting)
        return EC;
    }
    return std::error_code();
  }
  if (!::DeleteFileW(c_str(path_utf16))) {
    std::error_code EC = mapWindowsError(::GetLastError());
    if (EC != errc::no_such_file_or_directory || !IgnoreNonExisting)
      return EC;
  }
  return std::error_code();
}

std::error_code rename(const Twine &from, const Twine &to) {
  // Convert to utf-16.
  SmallVector<wchar_t, 128> wide_from;
  SmallVector<wchar_t, 128> wide_to;
  if (std::error_code ec = widenPath(from, wide_from))
    return ec;
  if (std::error_code ec = widenPath(to, wide_to))
    return ec;

  std::error_code ec = std::error_code();
  for (int i = 0; i < 2000; i++) {
    if (::MoveFileExW(wide_from.begin(), wide_to.begin(),
                      MOVEFILE_COPY_ALLOWED | MOVEFILE_REPLACE_EXISTING))
      return std::error_code();
    DWORD LastError = ::GetLastError();
    ec = mapWindowsError(LastError);
    if (LastError != ERROR_ACCESS_DENIED)
      break;
    // Retry MoveFile() at ACCESS_DENIED.
    // System scanners (eg. indexer) might open the source file when
    // It is written and closed.
    ::Sleep(1);
  }

  return ec;
}

std::error_code resize_file(int FD, uint64_t Size) {
#ifdef HAVE__CHSIZE_S
  errno_t error = ::_chsize_s(FD, Size);
#else
  errno_t error = ::_chsize(FD, Size);
#endif
  return std::error_code(error, std::generic_category());
}

std::error_code access(const Twine &Path, AccessMode Mode) {
  SmallVector<wchar_t, 128> PathUtf16;

  if (std::error_code EC = widenPath(Path, PathUtf16))
    return EC;

  DWORD Attributes = ::GetFileAttributesW(PathUtf16.begin());

  if (Attributes == INVALID_FILE_ATTRIBUTES) {
    // See if the file didn't actually exist.
    DWORD LastError = ::GetLastError();
    if (LastError != ERROR_FILE_NOT_FOUND &&
        LastError != ERROR_PATH_NOT_FOUND)
      return mapWindowsError(LastError);
    return errc::no_such_file_or_directory;
  }

  if (Mode == AccessMode::Write && (Attributes & FILE_ATTRIBUTE_READONLY))
    return errc::permission_denied;

  return std::error_code();
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

std::error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (std::error_code ec = status(A, fsA))
    return ec;
  if (std::error_code ec = status(B, fsB))
    return ec;
  result = equivalent(fsA, fsB);
  return std::error_code();
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

static std::error_code getStatus(HANDLE FileHandle, file_status &Result) {
  if (FileHandle == INVALID_HANDLE_VALUE)
    goto handle_status_error;

  switch (::GetFileType(FileHandle)) {
  default:
    llvm_unreachable("Don't know anything about this file type");
  case FILE_TYPE_UNKNOWN: {
    DWORD Err = ::GetLastError();
    if (Err != NO_ERROR)
      return mapWindowsError(Err);
    Result = file_status(file_type::type_unknown);
    return std::error_code();
  }
  case FILE_TYPE_DISK:
    break;
  case FILE_TYPE_CHAR:
    Result = file_status(file_type::character_file);
    return std::error_code();
  case FILE_TYPE_PIPE:
    Result = file_status(file_type::fifo_file);
    return std::error_code();
  }

  BY_HANDLE_FILE_INFORMATION Info;
  if (!::GetFileInformationByHandle(FileHandle, &Info))
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
    return std::error_code();
  }

handle_status_error:
  DWORD LastError = ::GetLastError();
  if (LastError == ERROR_FILE_NOT_FOUND ||
      LastError == ERROR_PATH_NOT_FOUND)
    Result = file_status(file_type::file_not_found);
  else if (LastError == ERROR_SHARING_VIOLATION)
    Result = file_status(file_type::type_unknown);
  else
    Result = file_status(file_type::status_error);
  return mapWindowsError(LastError);
}

std::error_code status(const Twine &path, file_status &result) {
  SmallString<128> path_storage;
  SmallVector<wchar_t, 128> path_utf16;

  StringRef path8 = path.toStringRef(path_storage);
  if (isReservedName(path8)) {
    result = file_status(file_type::character_file);
    return std::error_code();
  }

  if (std::error_code ec = widenPath(path8, path_utf16))
    return ec;

  DWORD attr = ::GetFileAttributesW(path_utf16.begin());
  if (attr == INVALID_FILE_ATTRIBUTES)
    return getStatus(INVALID_HANDLE_VALUE, result);

  // Handle reparse points.
  if (attr & FILE_ATTRIBUTE_REPARSE_POINT) {
    ScopedFileHandle h(
      ::CreateFileW(path_utf16.begin(),
                    0, // Attributes only.
                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS,
                    0));
    if (!h)
      return getStatus(INVALID_HANDLE_VALUE, result);
  }

  ScopedFileHandle h(
      ::CreateFileW(path_utf16.begin(), 0, // Attributes only.
                    FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0));
    if (!h)
      return getStatus(INVALID_HANDLE_VALUE, result);

    return getStatus(h, result);
}

std::error_code status(int FD, file_status &Result) {
  HANDLE FileHandle = reinterpret_cast<HANDLE>(_get_osfhandle(FD));
  return getStatus(FileHandle, Result);
}

std::error_code setLastModificationAndAccessTime(int FD, TimeValue Time) {
  ULARGE_INTEGER UI;
  UI.QuadPart = Time.toWin32Time();
  FILETIME FT;
  FT.dwLowDateTime = UI.LowPart;
  FT.dwHighDateTime = UI.HighPart;
  HANDLE FileHandle = reinterpret_cast<HANDLE>(_get_osfhandle(FD));
  if (!SetFileTime(FileHandle, NULL, &FT, &FT))
    return mapWindowsError(::GetLastError());
  return std::error_code();
}

std::error_code mapped_file_region::init(int FD, uint64_t Offset,
                                         mapmode Mode) {
  // Make sure that the requested size fits within SIZE_T.
  if (Size > std::numeric_limits<SIZE_T>::max())
    return make_error_code(errc::invalid_argument);

  HANDLE FileHandle = reinterpret_cast<HANDLE>(_get_osfhandle(FD));
  if (FileHandle == INVALID_HANDLE_VALUE)
    return make_error_code(errc::bad_file_descriptor);

  DWORD flprotect;
  switch (Mode) {
  case readonly:  flprotect = PAGE_READONLY; break;
  case readwrite: flprotect = PAGE_READWRITE; break;
  case priv:      flprotect = PAGE_WRITECOPY; break;
  }

  HANDLE FileMappingHandle =
      ::CreateFileMappingW(FileHandle, 0, flprotect,
                           (Offset + Size) >> 32,
                           (Offset + Size) & 0xffffffff,
                           0);
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
  Mapping = ::MapViewOfFile(FileMappingHandle,
                            dwDesiredAccess,
                            Offset >> 32,
                            Offset & 0xffffffff,
                            Size);
  if (Mapping == NULL) {
    std::error_code ec = mapWindowsError(GetLastError());
    ::CloseHandle(FileMappingHandle);
    return ec;
  }

  if (Size == 0) {
    MEMORY_BASIC_INFORMATION mbi;
    SIZE_T Result = VirtualQuery(Mapping, &mbi, sizeof(mbi));
    if (Result == 0) {
      std::error_code ec = mapWindowsError(GetLastError());
      ::UnmapViewOfFile(Mapping);
      ::CloseHandle(FileMappingHandle);
      return ec;
    }
    Size = mbi.RegionSize;
  }

  // Close all the handles except for the view. It will keep the other handles
  // alive.
  ::CloseHandle(FileMappingHandle);
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
    ::UnmapViewOfFile(Mapping);
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

std::error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                StringRef path){
  SmallVector<wchar_t, 128> path_utf16;

  if (std::error_code ec = widenPath(path, path_utf16))
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
  ScopedFindHandle FindHandle(::FindFirstFileW(c_str(path_utf16), &FirstFind));
  if (!FindHandle)
    return mapWindowsError(::GetLastError());

  size_t FilenameLen = ::wcslen(FirstFind.cFileName);
  while ((FilenameLen == 1 && FirstFind.cFileName[0] == L'.') ||
         (FilenameLen == 2 && FirstFind.cFileName[0] == L'.' &&
                              FirstFind.cFileName[1] == L'.'))
    if (!::FindNextFileW(FindHandle, &FirstFind)) {
      DWORD LastError = ::GetLastError();
      // Check for end.
      if (LastError == ERROR_NO_MORE_FILES)
        return detail::directory_iterator_destruct(it);
      return mapWindowsError(LastError);
    } else
      FilenameLen = ::wcslen(FirstFind.cFileName);

  // Construct the current directory entry.
  SmallString<128> directory_entry_name_utf8;
  if (std::error_code ec =
          UTF16ToUTF8(FirstFind.cFileName, ::wcslen(FirstFind.cFileName),
                      directory_entry_name_utf8))
    return ec;

  it.IterationHandle = intptr_t(FindHandle.take());
  SmallString<128> directory_entry_path(path);
  path::append(directory_entry_path, directory_entry_name_utf8);
  it.CurrentEntry = directory_entry(directory_entry_path);

  return std::error_code();
}

std::error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle != 0)
    // Closes the handle if it's valid.
    ScopedFindHandle close(HANDLE(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return std::error_code();
}

std::error_code detail::directory_iterator_increment(detail::DirIterState &it) {
  WIN32_FIND_DATAW FindData;
  if (!::FindNextFileW(HANDLE(it.IterationHandle), &FindData)) {
    DWORD LastError = ::GetLastError();
    // Check for end.
    if (LastError == ERROR_NO_MORE_FILES)
      return detail::directory_iterator_destruct(it);
    return mapWindowsError(LastError);
  }

  size_t FilenameLen = ::wcslen(FindData.cFileName);
  if ((FilenameLen == 1 && FindData.cFileName[0] == L'.') ||
      (FilenameLen == 2 && FindData.cFileName[0] == L'.' &&
                           FindData.cFileName[1] == L'.'))
    return directory_iterator_increment(it);

  SmallString<128> directory_entry_path_utf8;
  if (std::error_code ec =
          UTF16ToUTF8(FindData.cFileName, ::wcslen(FindData.cFileName),
                      directory_entry_path_utf8))
    return ec;

  it.CurrentEntry.replace_filename(Twine(directory_entry_path_utf8));
  return std::error_code();
}

std::error_code openFileForRead(const Twine &Name, int &ResultFD) {
  SmallVector<wchar_t, 128> PathUTF16;

  if (std::error_code EC = widenPath(Name, PathUTF16))
    return EC;

  HANDLE H = ::CreateFileW(PathUTF16.begin(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
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

  int FD = ::_open_osfhandle(intptr_t(H), 0);
  if (FD == -1) {
    ::CloseHandle(H);
    return mapWindowsError(ERROR_INVALID_HANDLE);
  }

  ResultFD = FD;
  return std::error_code();
}

std::error_code openFileForWrite(const Twine &Name, int &ResultFD,
                            sys::fs::OpenFlags Flags, unsigned Mode) {
  // Verify that we don't have both "append" and "excl".
  assert((!(Flags & sys::fs::F_Excl) || !(Flags & sys::fs::F_Append)) &&
         "Cannot specify both 'excl' and 'append' file creation flags!");

  SmallVector<wchar_t, 128> PathUTF16;

  if (std::error_code EC = widenPath(Name, PathUTF16))
    return EC;

  DWORD CreationDisposition;
  if (Flags & F_Excl)
    CreationDisposition = CREATE_NEW;
  else if (Flags & F_Append)
    CreationDisposition = OPEN_ALWAYS;
  else
    CreationDisposition = CREATE_ALWAYS;

  DWORD Access = GENERIC_WRITE;
  if (Flags & F_RW)
    Access |= GENERIC_READ;

  HANDLE H = ::CreateFileW(PathUTF16.begin(), Access,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                           CreationDisposition, FILE_ATTRIBUTE_NORMAL, NULL);

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

  int FD = ::_open_osfhandle(intptr_t(H), OpenFlags);
  if (FD == -1) {
    ::CloseHandle(H);
    return mapWindowsError(ERROR_INVALID_HANDLE);
  }

  ResultFD = FD;
  return std::error_code();
}
} // end namespace fs

namespace path {

bool home_directory(SmallVectorImpl<char> &result) {
#if 0 // HLSL Change Starts
  wchar_t Path[MAX_PATH];
  if (::SHGetFolderPathW(0, CSIDL_APPDATA | CSIDL_FLAG_CREATE, 0,
                         /*SHGFP_TYPE_CURRENT*/0, Path) != S_OK)
    return false;

  if (UTF16ToUTF8(Path, ::wcslen(Path), result))
    return false;

  return true;
#else
  return false;
#endif // HLSL Change Ends
}

static bool getTempDirEnvVar(const char *Var, SmallVectorImpl<char> &Res) {
  SmallVector<wchar_t, 128> NameUTF16;
  if (windows::UTF8ToUTF16(Var, NameUTF16))
    return false;

  SmallVector<wchar_t, 1024> Buf;
  size_t Size = 1024;
  do {
    Buf.reserve(Size);
    Size =
        GetEnvironmentVariableW(NameUTF16.data(), Buf.data(), Buf.capacity());
    if (Size == 0)
      return false;

    // Try again with larger buffer.
  } while (Size > Buf.capacity());
  Buf.set_size(Size);

  if (windows::UTF16ToUTF8(Buf.data(), Size, Res))
    return false;
  return true;
}

static bool getTempDirEnvVar(SmallVectorImpl<char> &Res) {
  const char *EnvironmentVariables[] = {"TMP", "TEMP", "USERPROFILE"};
  for (const char *Env : EnvironmentVariables) {
    if (getTempDirEnvVar(Env, Res))
      return true;
  }
  return false;
}

void system_temp_directory(bool ErasedOnReboot, SmallVectorImpl<char> &Result) {
  (void)ErasedOnReboot;
  Result.clear();

  // Check whether the temporary directory is specified by an environment
  // variable.
  if (getTempDirEnvVar(Result))
    return;

  // Fall back to a system default.
  const char *DefaultResult = "C:\\TEMP";
  Result.append(DefaultResult, DefaultResult + strlen(DefaultResult));
}
} // end namespace path

namespace windows {
std::error_code UTF8ToUTF16(llvm::StringRef utf8,
                            llvm::SmallVectorImpl<wchar_t> &utf16) {
  if (!utf8.empty()) {
    int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.begin(),
                                    utf8.size(), utf16.begin(), 0);

    if (len == 0)
      return mapWindowsError(::GetLastError());

    utf16.reserve(len + 1);
    utf16.set_size(len);

    len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.begin(),
                                utf8.size(), utf16.begin(), utf16.size());

    if (len == 0)
      return mapWindowsError(::GetLastError());
  }

  // Make utf16 null terminated.
  utf16.push_back(0);
  utf16.pop_back();

  return std::error_code();
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
