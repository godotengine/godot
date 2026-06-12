// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/file_util.h"

#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>  // close()
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

namespace libwebm {

std::string GetTempFileName() {
#if !defined _MSC_VER && !defined __MINGW32__
  std::string temp_file_name_template_str =
      std::string(std::getenv("TEST_TMPDIR") ? std::getenv("TEST_TMPDIR")
                                             : ".") +
      "/libwebm_temp.XXXXXX";
  char* temp_file_name_template =
      new char[temp_file_name_template_str.length() + 1];
  memset(temp_file_name_template, 0, temp_file_name_template_str.length() + 1);
  temp_file_name_template_str.copy(temp_file_name_template,
                                   temp_file_name_template_str.length(), 0);
  int fd = mkstemp(temp_file_name_template);
  std::string temp_file_name =
      (fd != -1) ? std::string(temp_file_name_template) : std::string();
  delete[] temp_file_name_template;
  if (fd != -1) {
    close(fd);
  }
  return temp_file_name;
#else
  char tmp_file_name[_MAX_PATH];
#if defined _MSC_VER || defined MINGW_HAS_SECURE_API
  errno_t err = tmpnam_s(tmp_file_name);
#else
  char* fname_pointer = tmpnam(tmp_file_name);
  int err = (fname_pointer == &tmp_file_name[0]) ? 0 : -1;
#endif
  if (err == 0) {
    return std::string(tmp_file_name);
  }
  return std::string();
#endif
}

uint64_t GetFileSize(const std::string& file_name) {
  uint64_t file_size = 0;
#ifndef _MSC_VER
  struct stat st;
  st.st_size = 0;
  if (stat(file_name.c_str(), &st) == 0) {
#else
  struct _stat st;
  st.st_size = 0;
  if (_stat(file_name.c_str(), &st) == 0) {
#endif
    file_size = st.st_size;
  }
  return file_size;
}

bool GetFileContents(const std::string& file_name, std::string* contents) {
  std::ifstream file(file_name.c_str());
  *contents = std::string(static_cast<size_t>(GetFileSize(file_name)), 0);
  if (file.good() && contents->size()) {
    file.read(&(*contents)[0], contents->size());
  }
  return !file.fail();
}

TempFileDeleter::TempFileDeleter() { file_name_ = GetTempFileName(); }

TempFileDeleter::~TempFileDeleter() {
  std::ifstream file(file_name_.c_str());
  if (file.good()) {
    file.close();
    std::remove(file_name_.c_str());
  }
}

}  // namespace libwebm
