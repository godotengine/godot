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

#ifndef CRASHPAD_UTIL_FILE_FILE_WRITER_H_
#define CRASHPAD_UTIL_FILE_FILE_WRITER_H_

#include <sys/types.h>

#include <vector>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/file/file_seeker.h"

namespace crashpad {

//! \brief A version of `iovec` with a `const` #iov_base field.
//!
//! This structure is intended to be used for write operations.
//
// Type compatibility with iovec is tested with static assertions in the
// implementation file.
struct WritableIoVec {
  //! \brief The base address of a memory region for output.
  const void* iov_base;

  //! \brief The size of the memory pointed to by #iov_base.
  size_t iov_len;
};

//! \brief An interface to write to files and other file-like objects with
//!     semantics matching the underlying platform (POSIX or Windows).
class FileWriterInterface : public virtual FileSeekerInterface {
 public:
  virtual ~FileWriterInterface() {}

  //! \brief Wraps LoggingWriteFile(), or provides an implementation with
  //!     identical semantics.
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged.
  virtual bool Write(const void* data, size_t size) = 0;

  //! \brief Wraps `writev()` on POSIX or provides an alternate implementation
  //!     with identical semantics. This method will write entire buffers,
  //!     continuing after a short write or after being interrupted. On
  //!     non-POSIX this is a simple wrapper around Write().
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged.
  //!
  //! \note The contents of \a iovecs are undefined when this method returns.
  virtual bool WriteIoVec(std::vector<WritableIoVec>* iovecs) = 0;
};

//! \brief A file writer backed by a FileHandle.
//!
//! FileWriter requires users to provide a FilePath to open, but this class
//! accepts an already-open FileHandle instead. Like FileWriter, this class may
//! write to a filesystem-based file, but unlike FileWriter, this class is not
//! responsible for creating or closing the file. Users of this class must
//! ensure that the file handle is closed appropriately elsewhere. Objects of
//! this class may be used to write to file handles not associated with
//! filesystem-based files, although special attention should be paid to the
//! Seek() method, which may not function on file handles that do not refer to
//! disk-based files.
//!
//! This class is expected to be used when other code is responsible for
//! creating files and already provides file handles.
class WeakFileHandleFileWriter : public FileWriterInterface {
 public:
  explicit WeakFileHandleFileWriter(FileHandle file_handle);
  ~WeakFileHandleFileWriter() override;

  // FileWriterInterface:
  bool Write(const void* data, size_t size) override;
  bool WriteIoVec(std::vector<WritableIoVec>* iovecs) override;

  // FileSeekerInterface:

  //! \copydoc FileWriterInterface::Seek()
  //!
  //! \note This method is only guaranteed to function on file handles referring
  //!     to disk-based files.
  FileOffset Seek(FileOffset offset, int whence) override;

 private:
  void set_file_handle(FileHandle file_handle) { file_handle_ = file_handle; }

  FileHandle file_handle_;  // weak

  // FileWriter uses this class as its internal implementation, and it needs to
  // be able to call set_file_handle(). FileWriter cannot initialize a
  // WeakFileHandleFileWriter with a correct file descriptor at the time of
  // construction because no file descriptor will be available until
  // FileWriter::Open() is called.
  friend class FileWriter;

  DISALLOW_COPY_AND_ASSIGN(WeakFileHandleFileWriter);
};

//! \brief A file writer implementation that wraps traditional system file
//!     operations on files accessed through the filesystem.
class FileWriter : public FileWriterInterface {
 public:
  FileWriter();
  ~FileWriter() override;

  // FileWriterInterface:

  //! \brief Wraps LoggingOpenFileForWrite().
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged.
  //!
  //! \note After a successful call, this method cannot be called again until
  //!     after Close().
  bool Open(const base::FilePath& path,
            FileWriteMode write_mode,
            FilePermissions permissions);

  //! \brief Wraps CheckedCloseHandle().
  //!
  //! \note It is only valid to call this method on an object that has had a
  //!     successful Open() that has not yet been matched by a subsequent call
  //!     to this method.
  void Close();

  // FileWriterInterface:

  //! \copydoc FileWriterInterface::Write()
  //!
  //! \note It is only valid to call this method between a successful Open() and
  //!     a Close().
  bool Write(const void* data, size_t size) override;

  //! \copydoc FileWriterInterface::WriteIoVec()
  //!
  //! \note It is only valid to call this method between a successful Open() and
  //!     a Close().
  bool WriteIoVec(std::vector<WritableIoVec>* iovecs) override;

  // FileSeekerInterface:

  //! \copydoc FileWriterInterface::Seek()
  //!
  //! \note It is only valid to call this method between a successful Open() and
  //!     a Close().
  FileOffset Seek(FileOffset offset, int whence) override;

 private:
  ScopedFileHandle file_;
  WeakFileHandleFileWriter weak_file_handle_file_writer_;

  DISALLOW_COPY_AND_ASSIGN(FileWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_FILE_WRITER_H_
