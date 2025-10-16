// Copyright 2015 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_FILE_INTERFACE_H_
#define _BSDIFF_FILE_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#include "bsdiff/common.h"

namespace bsdiff {

class BSDIFF_EXPORT FileInterface {
 public:
  virtual ~FileInterface() = default;

  // Reads synchronously from the current file position up to |count| bytes into
  // the passed |buf| buffer. On success, stores in |bytes_read| how many bytes
  // were actually read from the file. In case of error returns false. This
  // method may read less than |count| bytes even if there's no error. If the
  // end of file is reached, 0 bytes will be read and this methods returns true.
  virtual bool Read(void* buf, size_t count, size_t* bytes_read) = 0;

  // Writes synchronously up to |count| bytes from to passed |buf| buffer to
  // the file. On success, stores in |bytes_written| how many bytes
  // were actually written to the file. This method may write less than |count|
  // bytes and return successfully, or even write 0 bytes if there's no more
  // space left on the device. Returns whether the write succeeded.
  virtual bool Write(const void* buf, size_t count, size_t* bytes_written) = 0;

  // Change the current file position to |pos| bytes from the beginning of the
  // file. Return whether the seek succeeded.
  virtual bool Seek(int64_t pos) = 0;

  // Closes the file and flushes any cached data. Returns whether the close
  // succeeded.
  virtual bool Close() = 0;

  // Compute the size of the file and store it in |size|. Returns whether it
  // computed the size successfully.
  virtual bool GetSize(uint64_t* size) = 0;

 protected:
  FileInterface() = default;
};

}  // namespace bsdiff

#endif  // _BSDIFF_FILE_INTERFACE_H_
