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

#include "util/file/file_writer.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#include <algorithm>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "build/build_config.h"
#include "util/misc/implicit_cast.h"

#if defined(OS_POSIX)
#include <sys/uio.h>
#include <unistd.h>
#include "base/posix/eintr_wrapper.h"
#endif  // OS_POSIX

namespace crashpad {

#if defined(OS_POSIX)
// Ensure type compatibility between WritableIoVec and iovec.
static_assert(sizeof(WritableIoVec) == sizeof(iovec), "WritableIoVec size");
static_assert(offsetof(WritableIoVec, iov_base) == offsetof(iovec, iov_base),
              "WritableIoVec base offset");
static_assert(offsetof(WritableIoVec, iov_len) == offsetof(iovec, iov_len),
              "WritableIoVec len offset");
#endif  // OS_POSIX

WeakFileHandleFileWriter::WeakFileHandleFileWriter(FileHandle file_handle)
    : file_handle_(file_handle) {
}

WeakFileHandleFileWriter::~WeakFileHandleFileWriter() {
}

bool WeakFileHandleFileWriter::Write(const void* data, size_t size) {
  DCHECK_NE(file_handle_, kInvalidFileHandle);
  return LoggingWriteFile(file_handle_, data, size);
}

bool WeakFileHandleFileWriter::WriteIoVec(std::vector<WritableIoVec>* iovecs) {
  DCHECK_NE(file_handle_, kInvalidFileHandle);

  if (iovecs->empty()) {
    LOG(ERROR) << "WriteIoVec(): no iovecs";
    return false;
  }

#if defined(OS_POSIX)

  ssize_t size = 0;
  for (const WritableIoVec& iov : *iovecs) {
    // TODO(mark): Check to avoid overflow of ssize_t, and fail with EINVAL.
    size += iov.iov_len;
  }

  // Get an iovec*, because that’s what writev wants. The only difference
  // between WritableIoVec and iovec is that WritableIoVec’s iov_base is a
  // pointer to a const buffer, where iovec’s iov_base isn’t. writev doesn’t
  // actually write to the data, so this cast is safe here. iovec’s iov_base is
  // non-const because the same structure is used for readv and writev, and
  // readv needs to write to the buffer that iov_base points to.
  iovec* iov = reinterpret_cast<iovec*>(&(*iovecs)[0]);
  size_t remaining_iovecs = iovecs->size();

#if defined(OS_ANDROID)
  // Android does not expose the IOV_MAX macro, but makes its value available
  // via sysconf(). See Android 7.0.0 bionic/libc/bionic/sysconf.cpp sysconf().
  // Bionic defines IOV_MAX at bionic/libc/include/limits.h, but does not ship
  // this file to the NDK as <limits.h>, substituting
  // bionic/libc/include/bits/posix_limits.h.
  const size_t kIovMax = sysconf(_SC_IOV_MAX);
#else
  const size_t kIovMax = IOV_MAX;
#endif

  while (size > 0) {
    size_t writev_iovec_count = std::min(remaining_iovecs, kIovMax);
    ssize_t written =
        HANDLE_EINTR(writev(file_handle_, iov, writev_iovec_count));
    if (written < 0) {
      PLOG(ERROR) << "writev";
      return false;
    } else if (written == 0) {
      LOG(ERROR) << "writev: returned 0";
      return false;
    }

    size -= written;
    DCHECK_GE(size, 0);

    if (size == 0) {
      remaining_iovecs = 0;
      break;
    }

    while (written > 0) {
      size_t wrote_this_iovec =
          std::min(implicit_cast<size_t>(written), iov->iov_len);
      written -= wrote_this_iovec;
      if (wrote_this_iovec < iov->iov_len) {
        iov->iov_base =
            reinterpret_cast<char*>(iov->iov_base) + wrote_this_iovec;
        iov->iov_len -= wrote_this_iovec;
      } else {
        ++iov;
        --remaining_iovecs;
      }
    }
  }

  DCHECK_EQ(remaining_iovecs, 0u);

#else  // !OS_POSIX

  for (const WritableIoVec& iov : *iovecs) {
    if (!Write(iov.iov_base, iov.iov_len))
      return false;
  }

#endif  // OS_POSIX

#ifndef NDEBUG
  // The interface says that |iovecs| is not sacred, so scramble it to make sure
  // that nobody depends on it.
  memset(&(*iovecs)[0], 0xa5, sizeof((*iovecs)[0]) * iovecs->size());
#endif

  return true;
}

FileOffset WeakFileHandleFileWriter::Seek(FileOffset offset, int whence) {
  DCHECK_NE(file_handle_, kInvalidFileHandle);
  return LoggingSeekFile(file_handle_, offset, whence);
}

FileWriter::FileWriter()
    : file_(),
      weak_file_handle_file_writer_(kInvalidFileHandle) {
}

FileWriter::~FileWriter() {
}

bool FileWriter::Open(const base::FilePath& path,
                      FileWriteMode write_mode,
                      FilePermissions permissions) {
  CHECK(!file_.is_valid());
  file_.reset(LoggingOpenFileForWrite(path, write_mode, permissions));
  if (!file_.is_valid()) {
    return false;
  }

  weak_file_handle_file_writer_.set_file_handle(file_.get());
  return true;
}

void FileWriter::Close() {
  CHECK(file_.is_valid());

  weak_file_handle_file_writer_.set_file_handle(kInvalidFileHandle);
  file_.reset();
}

bool FileWriter::Write(const void* data, size_t size) {
  DCHECK(file_.is_valid());
  return weak_file_handle_file_writer_.Write(data, size);
}

bool FileWriter::WriteIoVec(std::vector<WritableIoVec>* iovecs) {
  DCHECK(file_.is_valid());
  return weak_file_handle_file_writer_.WriteIoVec(iovecs);
}

FileOffset FileWriter::Seek(FileOffset offset, int whence) {
  DCHECK(file_.is_valid());
  return weak_file_handle_file_writer_.Seek(offset, whence);
}

}  // namespace crashpad
