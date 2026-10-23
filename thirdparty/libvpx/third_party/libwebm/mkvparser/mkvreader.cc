// Copyright (c) 2010 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "mkvparser/mkvreader.h"

#include <sys/types.h>

#include <cassert>

namespace mkvparser {

MkvReader::MkvReader() : m_file(NULL), reader_owns_file_(true) {}

MkvReader::MkvReader(FILE* fp) : m_file(fp), reader_owns_file_(false) {
  GetFileSize();
}

MkvReader::~MkvReader() {
  if (reader_owns_file_)
    Close();
  m_file = NULL;
}

int MkvReader::Open(const char* fileName) {
  if (fileName == NULL)
    return -1;

  if (m_file)
    return -1;

#ifdef _MSC_VER
  const errno_t e = fopen_s(&m_file, fileName, "rb");

  if (e)
    return -1;  // error
#else
  m_file = fopen(fileName, "rb");

  if (m_file == NULL)
    return -1;
#endif
  return !GetFileSize();
}

bool MkvReader::GetFileSize() {
  if (m_file == NULL)
    return false;
#ifdef _MSC_VER
  int status = _fseeki64(m_file, 0L, SEEK_END);

  if (status)
    return false;  // error

  m_length = _ftelli64(m_file);
#else
  fseek(m_file, 0L, SEEK_END);
  m_length = ftell(m_file);
#endif
  assert(m_length >= 0);

  if (m_length < 0)
    return false;

#ifdef _MSC_VER
  status = _fseeki64(m_file, 0L, SEEK_SET);

  if (status)
    return false;  // error
#else
  fseek(m_file, 0L, SEEK_SET);
#endif

  return true;
}

void MkvReader::Close() {
  if (m_file != NULL) {
    fclose(m_file);
    m_file = NULL;
  }
}

int MkvReader::Length(long long* total, long long* available) {
  if (m_file == NULL)
    return -1;

  if (total)
    *total = m_length;

  if (available)
    *available = m_length;

  return 0;
}

int MkvReader::Read(long long offset, long len, unsigned char* buffer) {
  if (m_file == NULL)
    return -1;

  if (offset < 0)
    return -1;

  if (len < 0)
    return -1;

  if (len == 0)
    return 0;

  if (offset >= m_length)
    return -1;

#ifdef _MSC_VER
  const int status = _fseeki64(m_file, offset, SEEK_SET);

  if (status)
    return -1;  // error
#elif defined(_WIN32)
  fseeko64(m_file, static_cast<off_t>(offset), SEEK_SET);
#elif !(defined(__ANDROID__) && __ANDROID_API__ < 24 && !defined(__LP64__) && \
        defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64)
  // POSIX.1 has fseeko and ftello. fseeko and ftello are not available before
  // Android API level 24. See
  // https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md
  fseeko(m_file, static_cast<off_t>(offset), SEEK_SET);
#else
  fseek(m_file, static_cast<long>(offset), SEEK_SET);
#endif

  const size_t size = fread(buffer, 1, len, m_file);

  if (size < size_t(len))
    return -1;  // error

  return 0;  // success
}

}  // namespace mkvparser
