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

// Utility class for creating a temporary file for that is deleted in the
// destructor.

#ifndef COMMON_LINUX_SCOPED_TMPFILE_H_
#define COMMON_LINUX_SCOPED_TMPFILE_H_

#include <string>

namespace google_breakpad {

// Small RAII wrapper for temporary files.
//
// Example:
//   ScopedTmpFile tmp;
//   if (tmp.Init("Some file contents")) {
//     ...
//   }
class ScopedTmpFile {
 public:
  // Initialize the ScopedTmpFile object - this does not create the temporary
  // file until Init is called.
  ScopedTmpFile();

  // Destroy temporary file on scope exit.
  ~ScopedTmpFile();

  // Creates the empty temporary file - returns true iff the temporary file was
  // created successfully. Should always be checked before using the file.
  bool InitEmpty();

  // Creates the temporary file with the provided C string. The terminating null
  // is not written. Returns true iff the temporary file was created
  // successfully and the contents were written successfully.
  bool InitString(const char* text);

  // Creates the temporary file with the provided data. Returns true iff the
  // temporary file was created successfully and the contents were written
  // successfully.
  bool InitData(const void* data, size_t data_len);

  // Returns the Posix file descriptor for the test file, or -1 if Init()
  // returned false. Note: on Windows, this always returns -1.
  int GetFd() const {
    return fd_;
  }

 private:
  // Set the contents of the temporary file, and seek back to the start of the
  // file. On failure, returns false.
  bool SetContents(const void* data, size_t data_len);

  int fd_ = -1;
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_SCOPED_TMPFILE_H_
