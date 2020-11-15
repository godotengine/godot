// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/file/file_io.h"

#include <algorithm>
#include <limits>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/strings/utf_string_conversions.h"

namespace {

bool IsSocketHandle(HANDLE file) {
  if (GetFileType(file) == FILE_TYPE_PIPE) {
    // FILE_TYPE_PIPE means that it's a socket, a named pipe, or an anonymous
    // pipe. If we are unable to retrieve the pipe information, we know it's a
    // socket.
    return !GetNamedPipeInfo(file, NULL, NULL, NULL, NULL);
  }
  return false;
}

}  // namespace

namespace crashpad {

namespace {

// kMaxReadWriteSize needs to be limited to the range of DWORD for the calls to
// ::ReadFile() and ::WriteFile(), and also limited to the range of
// FileOperationResult to be able to adequately express the number of bytes read
// and written in the return values from ReadFile() and NativeWriteFile(). In a
// 64-bit build, the former will control, and the limit will be (2^32)-1. In a
// 32-bit build, the latter will control, and the limit will be (2^31)-1.
constexpr size_t kMaxReadWriteSize = std::min(
    static_cast<size_t>(std::numeric_limits<DWORD>::max()),
    static_cast<size_t>(std::numeric_limits<FileOperationResult>::max()));

FileHandle OpenFileForOutput(DWORD access,
                             const base::FilePath& path,
                             FileWriteMode mode,
                             FilePermissions permissions) {
  DCHECK(access & GENERIC_WRITE);
  DCHECK_EQ(access & ~(GENERIC_READ | GENERIC_WRITE), 0u);

  DWORD disposition = 0;
  switch (mode) {
    case FileWriteMode::kReuseOrFail:
      disposition = OPEN_EXISTING;
      break;
    case FileWriteMode::kReuseOrCreate:
      disposition = OPEN_ALWAYS;
      break;
    case FileWriteMode::kTruncateOrCreate:
      disposition = CREATE_ALWAYS;
      break;
    case FileWriteMode::kCreateOrFail:
      disposition = CREATE_NEW;
      break;
  }
  return CreateFile(path.value().c_str(),
                    access,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr,
                    disposition,
                    FILE_ATTRIBUTE_NORMAL,
                    nullptr);
}

}  // namespace

namespace internal {

FileOperationResult NativeWriteFile(FileHandle file,
                                    const void* buffer,
                                    size_t size) {
  // TODO(scottmg): This might need to handle the limit for pipes across a
  // network in the future.

  const DWORD write_size =
      static_cast<DWORD>(std::min(size, kMaxReadWriteSize));

  DWORD bytes_written;
  if (!::WriteFile(file, buffer, write_size, &bytes_written, nullptr))
    return -1;

  CHECK_NE(bytes_written, static_cast<DWORD>(-1));
  DCHECK_LE(static_cast<size_t>(bytes_written), write_size);
  return bytes_written;
}

}  // namespace internal

FileOperationResult ReadFile(FileHandle file, void* buffer, size_t size) {
  DCHECK(!IsSocketHandle(file));

  const DWORD read_size = static_cast<DWORD>(std::min(size, kMaxReadWriteSize));

  while (true) {
    DWORD bytes_read;
    BOOL success = ::ReadFile(file, buffer, read_size, &bytes_read, nullptr);
    if (!success) {
      if (GetLastError() == ERROR_BROKEN_PIPE) {
        // When reading a pipe and the write handle has been closed, ReadFile
        // fails with ERROR_BROKEN_PIPE, but only once all pending data has been
        // read. Treat this as EOF.
        return 0;
      }
      return -1;
    }

    CHECK_NE(bytes_read, static_cast<DWORD>(-1));
    DCHECK_LE(bytes_read, read_size);
    if (bytes_read != 0 || GetFileType(file) != FILE_TYPE_PIPE) {
      // Zero bytes read for a file indicates reaching EOF. Zero bytes read from
      // a pipe indicates only that there was a zero byte WriteFile issued on
      // the other end, so continue reading.
      return bytes_read;
    }
  }
}

FileHandle OpenFileForRead(const base::FilePath& path) {
  return CreateFile(path.value().c_str(),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr,
                    OPEN_EXISTING,
                    0,
                    nullptr);
}

FileHandle OpenFileForWrite(const base::FilePath& path,
                            FileWriteMode mode,
                            FilePermissions permissions) {
  return OpenFileForOutput(GENERIC_WRITE, path, mode, permissions);
}

FileHandle OpenFileForReadAndWrite(const base::FilePath& path,
                                   FileWriteMode mode,
                                   FilePermissions permissions) {
  return OpenFileForOutput(
      GENERIC_READ | GENERIC_WRITE, path, mode, permissions);
}

FileHandle LoggingOpenFileForRead(const base::FilePath& path) {
  FileHandle file = OpenFileForRead(path);
  PLOG_IF(ERROR, file == INVALID_HANDLE_VALUE)
      << "CreateFile " << base::UTF16ToUTF8(path.value());
  return file;
}

FileHandle LoggingOpenFileForWrite(const base::FilePath& path,
                                   FileWriteMode mode,
                                   FilePermissions permissions) {
  FileHandle file = OpenFileForWrite(path, mode, permissions);
  PLOG_IF(ERROR, file == INVALID_HANDLE_VALUE)
      << "CreateFile " << base::UTF16ToUTF8(path.value());
  return file;
}

FileHandle LoggingOpenFileForReadAndWrite(const base::FilePath& path,
                                          FileWriteMode mode,
                                          FilePermissions permissions) {
  FileHandle file = OpenFileForReadAndWrite(path, mode, permissions);
  PLOG_IF(ERROR, file == INVALID_HANDLE_VALUE)
      << "CreateFile " << base::UTF16ToUTF8(path.value());
  return file;
}

bool LoggingLockFile(FileHandle file, FileLocking locking) {
  DWORD flags =
      (locking == FileLocking::kExclusive) ? LOCKFILE_EXCLUSIVE_LOCK : 0;

  // Note that the `Offset` fields of overlapped indicate the start location for
  // locking (beginning of file in this case), and `hEvent` must be also be set
  // to 0.
  OVERLAPPED overlapped = {0};
  if (!LockFileEx(file, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
    PLOG(ERROR) << "LockFileEx";
    return false;
  }
  return true;
}

bool LoggingUnlockFile(FileHandle file) {
  // Note that the `Offset` fields of overlapped indicate the start location for
  // locking (beginning of file in this case), and `hEvent` must be also be set
  // to 0.
  OVERLAPPED overlapped = {0};
  if (!UnlockFileEx(file, 0, MAXDWORD, MAXDWORD, &overlapped)) {
    PLOG(ERROR) << "UnlockFileEx";
    return false;
  }
  return true;
}

FileOffset LoggingSeekFile(FileHandle file, FileOffset offset, int whence) {
  DWORD method = 0;
  switch (whence) {
    case SEEK_SET:
      method = FILE_BEGIN;
      break;
    case SEEK_CUR:
      method = FILE_CURRENT;
      break;
    case SEEK_END:
      method = FILE_END;
      break;
    default:
      NOTREACHED();
      break;
  }

  LARGE_INTEGER distance_to_move;
  distance_to_move.QuadPart = offset;
  LARGE_INTEGER new_offset;
  BOOL result = SetFilePointerEx(file, distance_to_move, &new_offset, method);
  if (!result) {
    PLOG(ERROR) << "SetFilePointerEx";
    return -1;
  }
  return new_offset.QuadPart;
}

bool LoggingTruncateFile(FileHandle file) {
  if (LoggingSeekFile(file, 0, SEEK_SET) != 0)
    return false;
  if (!SetEndOfFile(file)) {
    PLOG(ERROR) << "SetEndOfFile";
    return false;
  }
  return true;
}

bool LoggingCloseFile(FileHandle file) {
  BOOL rv = CloseHandle(file);
  PLOG_IF(ERROR, !rv) << "CloseHandle";
  return !!rv;
}

FileOffset LoggingFileSizeByHandle(FileHandle file) {
  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(file, &file_size)) {
    PLOG(ERROR) << "GetFileSizeEx";
    return -1;
  }
  return file_size.QuadPart;
}

FileHandle StdioFileHandle(StdioStream stdio_stream) {
  DWORD standard_handle;
  switch (stdio_stream) {
    case StdioStream::kStandardInput:
      standard_handle = STD_INPUT_HANDLE;
      break;
    case StdioStream::kStandardOutput:
      standard_handle = STD_OUTPUT_HANDLE;
      break;
    case StdioStream::kStandardError:
      standard_handle = STD_ERROR_HANDLE;
      break;
    default:
      NOTREACHED();
      return INVALID_HANDLE_VALUE;
  }

  HANDLE handle = GetStdHandle(standard_handle);
  PLOG_IF(ERROR, handle == INVALID_HANDLE_VALUE) << "GetStdHandle";
  return handle;
}

}  // namespace crashpad
