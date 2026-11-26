// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "mkvmuxer/mkvwriter.h"

#include <sys/types.h>

#ifdef _MSC_VER
#include <share.h>  // for _SH_DENYWR
#endif

namespace mkvmuxer {

MkvWriter::MkvWriter() : file_(NULL), writer_owns_file_(true) {}

MkvWriter::MkvWriter(FILE* fp) : file_(fp), writer_owns_file_(false) {}

MkvWriter::~MkvWriter() { Close(); }

int32 MkvWriter::Write(const void* buffer, uint32 length) {
  if (!file_)
    return -1;

  if (length == 0)
    return 0;

  if (buffer == NULL)
    return -1;

  const size_t bytes_written = fwrite(buffer, 1, length, file_);

  return (bytes_written == length) ? 0 : -1;
}

bool MkvWriter::Open(const char* filename) {
  if (filename == NULL)
    return false;

  if (file_)
    return false;

#ifdef _MSC_VER
  file_ = _fsopen(filename, "wb", _SH_DENYWR);
#else
  file_ = fopen(filename, "wb");
#endif
  if (file_ == NULL)
    return false;
  return true;
}

void MkvWriter::Close() {
  if (file_ && writer_owns_file_) {
    fclose(file_);
  }
  file_ = NULL;
}

int64 MkvWriter::Position() const {
  if (!file_)
    return 0;

#ifdef _MSC_VER
  return _ftelli64(file_);
#else
  return ftell(file_);
#endif
}

int32 MkvWriter::Position(int64 position) {
  if (!file_)
    return -1;

#ifdef _MSC_VER
  return _fseeki64(file_, position, SEEK_SET);
#elif defined(_WIN32)
  return fseeko64(file_, static_cast<off_t>(position), SEEK_SET);
#elif !(defined(__ANDROID__) && __ANDROID_API__ < 24 && !defined(__LP64__) && \
        defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64)
  // POSIX.1 has fseeko and ftello. fseeko and ftello are not available before
  // Android API level 24. See
  // https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md
  return fseeko(file_, static_cast<off_t>(position), SEEK_SET);
#else
  return fseek(file_, static_cast<long>(position), SEEK_SET);
#endif
}

bool MkvWriter::Seekable() const { return true; }

void MkvWriter::ElementStartNotify(uint64, int64) {}

}  // namespace mkvmuxer
