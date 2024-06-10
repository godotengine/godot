// Copyright 2022 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Utility class for creating a temporary file that is deleted in the
// destructor.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/linux/scoped_tmpfile.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/linux/eintr_wrapper.h"

#if !defined(__ANDROID__)
#define TEMPDIR "/tmp"
#else
#define TEMPDIR "/data/local/tmp"
#endif

namespace google_breakpad {

ScopedTmpFile::ScopedTmpFile() = default;

ScopedTmpFile::~ScopedTmpFile() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

bool ScopedTmpFile::InitEmpty() {
  // Prevent calling Init when fd_ is already valid, leaking the file.
  if (fd_ != -1) {
    return false;
  }

  // Respect the TMPDIR environment variable.
  const char* tempdir = getenv("TMPDIR");
  if (!tempdir) {
    tempdir = TEMPDIR;
  }

  // Create a temporary file that is not linked in to the filesystem, and that
  // is only accessible by the current user.
  fd_ = open(tempdir, O_TMPFILE | O_RDWR, S_IRUSR | S_IWUSR);

  return fd_ >= 0;
}

bool ScopedTmpFile::InitString(const char* text) {
  return InitData(text, strlen(text));
}

bool ScopedTmpFile::InitData(const void* data, size_t data_len) {
  if (!InitEmpty()) {
    return false;
  }

  return SetContents(data, data_len);
}

bool ScopedTmpFile::SetContents(const void* data, size_t data_len) {
  ssize_t r = HANDLE_EINTR(write(fd_, data, data_len));
  if (r != static_cast<ssize_t>(data_len)) {
    return false;
  }

  return 0 == lseek(fd_, 0, SEEK_SET);
}

}  // namespace google_breakpad
