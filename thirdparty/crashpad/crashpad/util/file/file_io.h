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

#ifndef CRASHPAD_UTIL_FILE_FILE_IO_H_
#define CRASHPAD_UTIL_FILE_FILE_IO_H_

#include <sys/types.h>

#include <string>

#include "build/build_config.h"

#if defined(OS_POSIX)
#include "base/files/scoped_file.h"
#elif defined(OS_WIN)
#include <windows.h>
#include "util/win/scoped_handle.h"
#endif

namespace base {
class FilePath;
}  // namespace base

namespace crashpad {

#if defined(OS_POSIX) || DOXYGEN

//! \brief Platform-specific alias for a low-level file handle.
using FileHandle = int;

//! \brief Platform-specific alias for a position in an open file.
using FileOffset = off_t;

//! \brief Scoped wrapper of a FileHandle.
using ScopedFileHandle = base::ScopedFD;

//! \brief The return value of read and write calls.
using FileOperationResult = ssize_t;

//! \brief A value that can never be a valid FileHandle.
const FileHandle kInvalidFileHandle = -1;

#elif defined(OS_WIN)

using FileHandle = HANDLE;
using FileOffset = LONGLONG;
using ScopedFileHandle = ScopedFileHANDLE;
using FileOperationResult = LONG_PTR;

const FileHandle kInvalidFileHandle = INVALID_HANDLE_VALUE;

#endif

//! \brief Determines the mode that LoggingOpenFileForWrite() uses.
enum class FileWriteMode {
  //! \brief Opens the file if it exists, or fails if it does not.
  kReuseOrFail,

  //! \brief Opens the file if it exists, or creates a new file.
  kReuseOrCreate,

  //! \brief Creates a new file. If the file already exists, it will be
  //!     overwritten.
  kTruncateOrCreate,

  //! \brief Creates a new file. If the file already exists, the open will fail.
  kCreateOrFail,
};

//! \brief Determines the permissions bits for files created on POSIX systems.
enum class FilePermissions : bool {
  //! \brief Equivalent to `0600`.
  kOwnerOnly,

  //! \brief Equivalent to `0644`.
  kWorldReadable,
};

//! \brief Determines the locking mode that LoggingLockFile() uses.
enum class FileLocking : bool {
  //! \brief Equivalent to `flock()` with `LOCK_SH`.
  kShared,

  //! \brief Equivalent to `flock()` with `LOCK_EX`.
  kExclusive,
};

//! \brief Determines the FileHandle that StdioFileHandle() returns.
enum class StdioStream {
  //! \brief Standard input, or `stdin`.
  kStandardInput,

  //! \brief Standard output, or `stdout`.
  kStandardOutput,

  //! \brief Standard error, or `stderr`.
  kStandardError,
};

namespace internal {

#if defined(OS_POSIX) || DOXYGEN

//! \brief The name of the native read function used by ReadFile().
//!
//! This value may be useful for logging.
//!
//! \sa kNativeWriteFunctionName
constexpr char kNativeReadFunctionName[] = "read";

//! \brief The name of the native write function used by WriteFile().
//!
//! This value may be useful for logging.
//!
//! \sa kNativeReadFunctionName
constexpr char kNativeWriteFunctionName[] = "write";

#elif defined(OS_WIN)

constexpr char kNativeReadFunctionName[] = "ReadFile";
constexpr char kNativeWriteFunctionName[] = "WriteFile";

#endif

//! \brief The internal implementation of ReadFileExactly() and its wrappers.
//!
//! The logic is exposed so that it may be reused by FileReaderInterface, and
//! so that it may be tested without requiring large files to be read. It is not
//! intended to be used more generally. Use ReadFileExactly(),
//! LoggingReadFileExactly(), CheckedReadFileExactly(), or
//! FileReaderInterface::ReadExactly() instead.
class ReadExactlyInternal {
 public:
  //! \brief Calls Read(), retrying following a short read, ensuring that
  //!     exactly \a size bytes are read.
  //!
  //! \return `true` on success. `false` if the underlying Read() fails or if
  //!     fewer than \a size bytes were read. When returning `false`, if \a
  //!     can_log is `true`, logs a message.
  bool ReadExactly(void* buffer, size_t size, bool can_log);

 protected:
  ReadExactlyInternal() {}
  ~ReadExactlyInternal() {}

 private:
  //! \brief Wraps a read operation, such as ReadFile().
  //!
  //! \return The number of bytes read and placed into \a buffer, or `-1` on
  //!     error. When returning `-1`, if \a can_log is `true`, logs a message.
  virtual FileOperationResult Read(void* buffer, size_t size, bool can_log) = 0;

  DISALLOW_COPY_AND_ASSIGN(ReadExactlyInternal);
};

//! \brief The internal implementation of WriteFile() and its wrappers.
//!
//! The logic is exposed so that it may be tested without requiring large files
//! to be written. It is not intended to be used more generally. Use
//! WriteFile(), LoggingWriteFile(), CheckedWriteFile(), or
//! FileWriterInterface::Write() instead.
class WriteAllInternal {
 public:
  //! \brief Calls Write(), retrying following a short write, ensuring that
  //!     exactly \a size bytes are written.
  //!
  //! \return `true` on success. `false` if the underlying Write() fails or if
  //!     fewer than \a size bytes were written.
  bool WriteAll(const void* buffer, size_t size);

 protected:
  WriteAllInternal() {}
  ~WriteAllInternal() {}

 private:
  //! \brief Wraps a write operation, such as NativeWriteFile().
  //!
  //! \return The number of bytes written from \a buffer, or `-1` on error.
  virtual FileOperationResult Write(const void* buffer, size_t size) = 0;

  DISALLOW_COPY_AND_ASSIGN(WriteAllInternal);
};

//! \brief Writes to a file, retrying when interrupted on POSIX.
//!
//! Fewer than \a size bytes may be written to \a file. This can happen if the
//! underlying write operation returns before writing the entire buffer, or if
//! the buffer is too large to write in a single operation, possibly due to a
//! limitation of a data type used to express the number of bytes written.
//!
//! This function adapts native write operations for uniform use by WriteFile().
//! This function should only be called by WriteFile(). Other code should call
//! WriteFile() or another function that wraps WriteFile().
//!
//! \param[in] file The file to write to.
//! \param[in] buffer A buffer containing data to be written.
//! \param[in] size The number of bytes from \a buffer to write.
//!
//! \return The number of bytes actually written from \a buffer to \a file on
//!     success. `-1` on error, with `errno` or `GetLastError()` set
//!     appropriately.
FileOperationResult NativeWriteFile(FileHandle file,
                                    const void* buffer,
                                    size_t size);

}  // namespace internal

//! \brief Reads from a file, retrying when interrupted before reading any data
//!     on POSIX.
//!
//! This function reads into \a buffer. Fewer than \a size bytes may be read.
//! On Windows, reading from sockets is not currently supported.
//!
//! \return The number of bytes read and placed into \a buffer, or `-1` on
//!     error, with `errno` or `GetLastError()` set appropriately. On error, a
//!     portion of \a file may have been read into \a buffer.
//!
//! \sa WriteFile
//! \sa ReadFileExactly
//! \sa LoggingReadFileExactly
//! \sa CheckedReadFileExactly
//! \sa CheckedReadFileAtEOF
FileOperationResult ReadFile(FileHandle file, void* buffer, size_t size);

//! \brief Writes to a file, retrying when interrupted on POSIX or following a
//!     short write.
//!
//! This function writes to \a file, stopping only when \a size bytes have been
//! written.
//!
//! \return `true` on success. `false` on error, with `errno` or
//!     `GetLastError()` set appropriately. On error, a portion of \a buffer may
//!     have been written to \a file.
//!
//! \sa ReadFile
//! \sa LoggingWriteFile
//! \sa CheckedWriteFile
bool WriteFile(FileHandle file, const void* buffer, size_t size);

//! \brief Wraps ReadFile(), retrying following a short read, ensuring that
//!     exactly \a size bytes are read.
//!
//! \return `true` on success. If the underlying ReadFile() fails, or if fewer
//!     than \a size bytes were read, this function logs a message and
//!     returns `false`.
//!
//! \sa LoggingWriteFile
//! \sa ReadFile
//! \sa LoggingReadFileExactly
//! \sa CheckedReadFileExactly
//! \sa CheckedReadFileAtEOF
bool ReadFileExactly(FileHandle file, void* buffer, size_t size);

//! \brief Wraps ReadFile(), retrying following a short read, ensuring that
//!     exactly \a size bytes are read.
//!
//! \return `true` on success. If the underlying ReadFile() fails, or if fewer
//!     than \a size bytes were read, this function logs a message and
//!     returns `false`.
//!
//! \sa LoggingWriteFile
//! \sa ReadFile
//! \sa ReadFileExactly
//! \sa CheckedReadFileExactly
//! \sa CheckedReadFileAtEOF
bool LoggingReadFileExactly(FileHandle file, void* buffer, size_t size);

//! \brief Wraps WriteFile(), ensuring that exactly \a size bytes are written.
//!
//! \return `true` on success. If the underlying WriteFile() fails, or if fewer
//!     than \a size bytes were written, this function logs a message and
//!     returns `false`.
//!
//! \sa LoggingReadFileExactly
//! \sa WriteFile
//! \sa CheckedWriteFile
bool LoggingWriteFile(FileHandle file, const void* buffer, size_t size);

//! \brief Wraps ReadFile(), ensuring that exactly \a size bytes are read.
//!
//! If the underlying ReadFile() fails, or if fewer than \a size bytes were
//! read, this function causes execution to terminate without returning.
//!
//! \sa CheckedWriteFile
//! \sa ReadFile
//! \sa LoggingReadFileExactly
//! \sa CheckedReadFileAtEOF
void CheckedReadFileExactly(FileHandle file, void* buffer, size_t size);

//! \brief Wraps WriteFile(), ensuring that exactly \a size bytes are written.
//!
//! if the underlying WriteFile() fails, or if fewer than \a size bytes were
//! written, this function causes execution to terminate without returning.
//!
//! \sa CheckedReadFileExactly
//! \sa WriteFile
//! \sa LoggingWriteFile
void CheckedWriteFile(FileHandle file, const void* buffer, size_t size);

//! \brief Wraps ReadFile(), ensuring that it indicates end-of-file.
//!
//! Attempts to read a single byte from \a file, expecting no data to be read.
//! If the underlying ReadFile() fails, or if a byte actually is read, this
//! function causes execution to terminate without returning.
//!
//! \sa CheckedReadFileExactly
//! \sa ReadFile
void CheckedReadFileAtEOF(FileHandle file);

//! \brief Wraps ReadFile() to read from the current file position to the end of
//!     the file into \a contents.
//!
//! \return `true` on success, or `false` with a message logged.
bool LoggingReadToEOF(FileHandle file, std::string* contents);

//! \brief Wraps LoggingOpenFileForRead() and ReadFile() reading the entire file
//!     into \a contents.
//!
//! \return `true` on success, or `false` with a message logged.
bool LoggingReadEntireFile(const base::FilePath& path, std::string* contents);

//! \brief Wraps `open()` or `CreateFile()`, opening an existing file for
//!     reading.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa OpenFileForWrite
//! \sa OpenFileForReadAndWrite
//! \sa LoggingOpenFileForRead
FileHandle OpenFileForRead(const base::FilePath& path);

//! \brief Wraps `open()` or `CreateFile()`, creating a file for output.
//!
//! \a mode determines the style (truncate, reuse, etc.) that is used to open
//! the file. On POSIX, \a permissions determines the value that is passed as
//! `mode` to `open()`. On Windows, the file is always opened in binary mode
//! (that is, no CRLF translation). On Windows, the file is opened for sharing,
//! see LoggingLockFile() and LoggingUnlockFile() to control concurrent access.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa OpenFileForRead
//! \sa OpenFileForReadAndWrite
//! \sa LoggingOpenFileForWrite
FileHandle OpenFileForWrite(const base::FilePath& path,
                            FileWriteMode mode,
                            FilePermissions permissions);

//! \brief Wraps `open()` or `CreateFile()`, creating a file for both input and
//!     output.
//!
//! \a mode determines the style (truncate, reuse, etc.) that is used to open
//! the file. On POSIX, \a permissions determines the value that is passed as
//! `mode` to `open()`. On Windows, the file is always opened in binary mode
//! (that is, no CRLF translation). On Windows, the file is opened for sharing,
//! see LoggingLockFile() and LoggingUnlockFile() to control concurrent access.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa OpenFileForRead
//! \sa OpenFileForWrite
//! \sa LoggingOpenFileForReadAndWrite
FileHandle OpenFileForReadAndWrite(const base::FilePath& path,
                                   FileWriteMode mode,
                                   FilePermissions permissions);

//! \brief Wraps OpenFileForRead(), logging an error if the operation fails.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa LoggingOpenFileForWrite
//! \sa LoggingOpenFileForReadAndWrite
FileHandle LoggingOpenFileForRead(const base::FilePath& path);

//! \brief Wraps OpenFileForWrite(), logging an error if the operation fails.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa LoggingOpenFileForRead
//! \sa LoggingOpenFileForReadAndWrite
FileHandle LoggingOpenFileForWrite(const base::FilePath& path,
                                   FileWriteMode mode,
                                   FilePermissions permissions);

//! \brief Wraps OpenFileForReadAndWrite(), logging an error if the operation
//!     fails.
//!
//! \return The newly opened FileHandle, or an invalid FileHandle on failure.
//!
//! \sa ScopedFileHandle
//! \sa LoggingOpenFileForRead
//! \sa LoggingOpenFileForWrite
FileHandle LoggingOpenFileForReadAndWrite(const base::FilePath& path,
                                          FileWriteMode mode,
                                          FilePermissions permissions);

// Fuchsia does not currently support any sort of file locking. See
// https://crashpad.chromium.org/bug/196 and
// https://crashpad.chromium.org/bug/217.
#if !defined(OS_FUCHSIA)

//! \brief Locks the given \a file using `flock()` on POSIX or `LockFileEx()` on
//!     Windows.
//!
//! It is an error to attempt to lock a file in a different mode when it is
//! already locked. This call will block until the lock is acquired. The
//! entire file is locked.
//!
//! If \a locking is FileLocking::kShared, \a file must have been opened for
//! reading, and if it's FileLocking::kExclusive, \a file must have been opened
//! for writing.
//!
//! \param[in] file The open file handle to be locked.
//! \param[in] locking Controls whether the lock is a shared reader lock, or an
//!     exclusive writer lock.
//!
//! \return `true` on success, or `false` and a message will be logged.
bool LoggingLockFile(FileHandle file, FileLocking locking);

//! \brief Unlocks a file previously locked with LoggingLockFile().
//!
//! It is an error to attempt to unlock a file that was not previously locked.
//! A previously-locked file should be unlocked before closing the file handle,
//! otherwise on some OSs the lock may not be released immediately.
//!
//! \param[in] file The open locked file handle to be unlocked.
//!
//! \return `true` on success, or `false` and a message will be logged.
bool LoggingUnlockFile(FileHandle file);

#endif  // !OS_FUCHSIA

//! \brief Wraps `lseek()` or `SetFilePointerEx()`. Logs an error if the
//!     operation fails.
//!
//! Repositions the offset of the open \a file to the specified \a offset,
//! relative to \a whence. \a whence must be one of `SEEK_SET`, `SEEK_CUR`, or
//! `SEEK_END`, and is interpreted in the usual way.
//!
//! \return The resulting offset in bytes from the beginning of the file, or
//!     `-1` on failure.
FileOffset LoggingSeekFile(FileHandle file, FileOffset offset, int whence);

//! \brief Truncates the given \a file to zero bytes in length.
//!
//! \return `true` on success, or `false`, and a message will be logged.
bool LoggingTruncateFile(FileHandle file);

//! \brief Wraps `close()` or `CloseHandle()`, logging an error if the operation
//!     fails.
//!
//! \return On success, `true` is returned. On failure, an error is logged and
//!     `false` is returned.
bool LoggingCloseFile(FileHandle file);

//! \brief Wraps `close()` or `CloseHandle()`, ensuring that it succeeds.
//!
//! If the underlying function fails, this function causes execution to
//! terminate without returning.
void CheckedCloseFile(FileHandle file);

//! \brief Determines the size of a file.
//!
//! \param[in] file The handle to the file for which the size should be
//!     retrieved.
//!
//! \return The size of the file. If an error occurs when attempting to
//!     determine its size, returns `-1` with an error logged.
FileOffset LoggingFileSizeByHandle(FileHandle file);

//! \brief Returns a FileHandle corresponding to the requested standard I/O
//!     stream.
//!
//! The returned FileHandle should not be closed on POSIX, where it is
//! important to maintain valid file descriptors occupying the slots reserved
//! for these streams. If a need to close such a stream arises on POSIX,
//! `dup2()` should instead be used to replace the existing file descriptor with
//! one opened to `/dev/null`. See CloseStdinAndStdout().
//!
//! \param[in] stdio_stream The requested standard I/O stream.
//!
//! \return A corresponding FileHandle on success. kInvalidFileHandle on error,
//!     with a message logged.
FileHandle StdioFileHandle(StdioStream stdio_stream);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_FILE_IO_H_
