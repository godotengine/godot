// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_COMMON_FILE_UTIL_H_
#define LIBWEBM_COMMON_FILE_UTIL_H_

#include <stdint.h>

#include <string>

#include "mkvmuxer/mkvmuxertypes.h"  // LIBWEBM_DISALLOW_COPY_AND_ASSIGN()

namespace libwebm {

// Returns a temporary file name.
std::string GetTempFileName();

// Returns size of file specified by |file_name|, or 0 upon failure.
uint64_t GetFileSize(const std::string& file_name);

// Gets the contents file_name as a string. Returns false on error.
bool GetFileContents(const std::string& file_name, std::string* contents);

// Manages life of temporary file specified at time of construction. Deletes
// file upon destruction.
class TempFileDeleter {
 public:
  TempFileDeleter();
  explicit TempFileDeleter(std::string file_name) : file_name_(file_name) {}
  ~TempFileDeleter();
  const std::string& name() const { return file_name_; }

 private:
  std::string file_name_;
  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(TempFileDeleter);
};

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_FILE_UTIL_H_
