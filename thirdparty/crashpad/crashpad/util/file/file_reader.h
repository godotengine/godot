// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_FILE_FILE_READER_H_
#define CRASHPAD_UTIL_FILE_FILE_READER_H_

#include <sys/types.h>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/file/file_seeker.h"

namespace crashpad {

//! \brief An interface to read to files and other file-like objects with
//!     semantics matching the underlying platform (POSIX or Windows).
class FileReaderInterface : public virtual FileSeekerInterface {
 public:
  virtual ~FileReaderInterface() {}

  //! \brief Wraps ReadFile(), or provides an implementation with identical
  //!     semantics.
  //!
  //! \return The number of bytes actually read if the operation succeeded,
  //!     which may be `0` or any positive value less than or equal to \a size.
  //!     `-1` if the operation failed, with an error message logged.
  virtual FileOperationResult Read(void* data, size_t size) = 0;

  //! \brief Wraps Read(), ensuring that the read succeeded and exactly \a size
  //!     bytes were read.
  //!
  //! Semantically, this behaves as LoggingReadFileExactly().
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged. Short reads are treated as failures.
  bool ReadExactly(void* data, size_t size);
};

//! \brief A file reader backed by a FileHandle.
//!
//! FileReader requires users to provide a FilePath to open, but this class
//! accepts an already-open FileHandle instead. Like FileReader, this class may
//! read from a filesystem-based file, but unlike FileReader, this class is not
//! responsible for opening or closing the file. Users of this class must ensure
//! that the file handle is closed appropriately elsewhere. Objects of this
//! class may be used to read from file handles not associated with
//! filesystem-based files, although special attention should be paid to the
//! Seek() method, which may not function on file handles that do not refer to
//! disk-based files.
//!
//! This class is expected to be used when other code is responsible for
//! opening files and already provides file handles.
class WeakFileHandleFileReader : public FileReaderInterface {
 public:
  explicit WeakFileHandleFileReader(FileHandle file_handle);
  ~WeakFileHandleFileReader() override;

  // FileReaderInterface:
  FileOperationResult Read(void* data, size_t size) override;

  // FileSeekerInterface:

  //! \copydoc FileReaderInterface::Seek()
  //!
  //! \note This method is only guaranteed to function on file handles referring
  //!     to disk-based files.
  FileOffset Seek(FileOffset offset, int whence) override;

 private:
  void set_file_handle(FileHandle file_handle) { file_handle_ = file_handle; }

  FileHandle file_handle_;  // weak

  // FileReader uses this class as its internal implementation, and it needs to
  // be able to call set_file_handle(). FileReader cannot initialize a
  // WeakFileHandleFileReader with a correct file descriptor at the time of
  // construction because no file descriptor will be available until
  // FileReader::Open() is called.
  friend class FileReader;

  DISALLOW_COPY_AND_ASSIGN(WeakFileHandleFileReader);
};

//! \brief A file reader implementation that wraps traditional system file
//!     operations on files accessed through the filesystem.
class FileReader : public FileReaderInterface {
 public:
  FileReader();
  ~FileReader() override;

  // FileReaderInterface:

  //! \brief Wraps LoggingOpenFileForRead().
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged.
  //!
  //! \note After a successful call, this method cannot be called again until
  //!     after Close().
  bool Open(const base::FilePath& path);

  //! \brief Wraps CheckedCloseHandle().
  //!
  //! \note It is only valid to call this method on an object that has had a
  //!     successful Open() that has not yet been matched by a subsequent call
  //!     to this method.
  void Close();

  // FileReaderInterface:

  //! \copydoc FileReaderInterface::Read()
  //!
  //! \note It is only valid to call this method between a successful Open() and
  //!     a Close().
  FileOperationResult Read(void* data, size_t size) override;

  // FileSeekerInterface:

  //! \copydoc FileReaderInterface::Seek()
  //!
  //! \note It is only valid to call this method between a successful Open() and
  //!     a Close().
  FileOffset Seek(FileOffset offset, int whence) override;

 private:
  ScopedFileHandle file_;
  WeakFileHandleFileReader weak_file_handle_file_reader_;

  DISALLOW_COPY_AND_ASSIGN(FileReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_FILE_READER_H_
